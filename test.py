import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.common.transformer import Transformer
from protomotions.agents.masked_mimic.model import VaeDeterministicOutputModel
from protomotions.utils import config_utils  # This registers the resolvers
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree

def summarize_tree(d):
    if hasattr(d, "shape"):
        return d.shape
    elif isinstance(d, dict):
        return {k:summarize_tree(v) for k, v in d.items()}
    else:
        return d

def quat_to_6d(quat):
    """Convert quaternion to 6D rotation representation (first two columns of rotation matrix)"""
    # Normalize quaternion
    quat = quat / torch.norm(quat)
    
    # Convert to rotation matrix
    w, x, y, z = quat[3], quat[0], quat[1], quat[2]  # Note: quat = [x,y,z,w]
    
    rot_mat = torch.zeros((3, 3))
    
    # First column
    rot_mat[0, 0] = 1 - 2*y*y - 2*z*z
    rot_mat[1, 0] = 2*x*y + 2*w*z
    rot_mat[2, 0] = 2*x*z - 2*w*y
    
    # Second column
    rot_mat[0, 1] = 2*x*y - 2*w*z
    rot_mat[1, 1] = 1 - 2*x*x - 2*z*z
    rot_mat[2, 1] = 2*y*z + 2*w*x
    
    # Extract first two columns and flatten
    return rot_mat[:, :2].reshape(-1)

def quat_to_matrix(quat):
    """Convert quaternion to rotation matrix"""
    # Normalize quaternion
    quat = quat / torch.norm(quat)
    
    # Convert to rotation matrix
    w, x, y, z = quat[3], quat[0], quat[1], quat[2]  # Note: quat = [x,y,z,w]
    
    rot_mat = torch.zeros((3, 3))
    
    # First row
    rot_mat[0, 0] = 1 - 2*y*y - 2*z*z
    rot_mat[0, 1] = 2*x*y - 2*w*z
    rot_mat[0, 2] = 2*x*z + 2*w*y
    
    # Second row
    rot_mat[1, 0] = 2*x*y + 2*w*z
    rot_mat[1, 1] = 1 - 2*x*x - 2*z*z
    rot_mat[1, 2] = 2*y*z - 2*w*x
    
    # Third row
    rot_mat[2, 0] = 2*x*z - 2*w*y
    rot_mat[2, 1] = 2*y*z + 2*w*x
    rot_mat[2, 2] = 1 - 2*x*x - 2*y*y
    
    return rot_mat

def compute_body_positions(motion_data, frame_idx):
    """Compute body positions using forward kinematics"""
    # Get rotations and root position
    rotations = motion_data['rotation'][frame_idx]  # [24, 4]
    root_pos = motion_data['root_translation'][frame_idx]  # [3]
    skeleton = motion_data['skeleton_tree']
    
    # Initialize body positions
    body_pos = torch.zeros((24, 3))  # 24 bodies, 3 coordinates each
    body_pos[0] = root_pos  # Root position
    
    # Convert rotations to matrices
    rot_matrices = torch.stack([quat_to_matrix(quat) for quat in rotations])  # [24, 3, 3]
    
    # Forward kinematics
    for joint_name, joint_info in skeleton.items():
        if joint_name == 'pelvis':  # Root joint
            continue
            
        # Get joint index and parent index
        joint_idx = joint_info['index']
        parent_idx = skeleton[joint_info['parent']]['index']
        
        # Get local offset from parent
        offset = torch.tensor(joint_info['offset'], dtype=torch.float32)
        
        # Transform offset by parent rotation
        global_offset = rot_matrices[parent_idx] @ offset
        
        # Add to parent position
        body_pos[joint_idx] = body_pos[parent_idx] + global_offset
    
    return body_pos

def construct_pose_obs(motion: SkeletonMotion, frame_idx: int, add_time: bool = False):
    """
    Construct pose observation from motion data at given frame
    Returns observation vector matching the model's expected format
    
    Args:
        motion: SkeletonMotion object
        frame_idx: Frame index to get pose from
        add_time: Whether to add time dimension (for future poses)
    """
    # Get root position and height
    root_pos = motion.root_translation[frame_idx]
    ground_height = torch.tensor(0.0)
    root_h = root_pos[2:3] - ground_height
    
    # Get local body positions relative to root (69-dim = 23 bodies * 3 coords)
    # Exclude root position
    body_pos = motion.global_translation[frame_idx]  # [24, 3]
    local_body_pos = body_pos[1:] - root_pos.unsqueeze(0)  # [23, 3]
    local_body_pos_flat = local_body_pos.reshape(-1)  # [69]
    
    # Get local body rotations (144-dim = 24 bodies * 6 coords)
    # Convert quaternions to 6D rotation representation
    body_rots = motion.rotation[frame_idx]  # [24, 4]
    body_rot_6d = torch.cat([quat_to_6d(rot) for rot in body_rots])  # [144]
    
    # Get velocities (72-dim = 24 bodies * 3 coords each)
    body_vel = motion.global_velocity[frame_idx].reshape(-1)  # [72]
    body_ang_vel = motion.global_angular_velocity[frame_idx].reshape(-1)  # [72]
    
    # Concatenate all components
    obs = torch.cat([
        root_h,
        local_body_pos_flat,
        body_rot_6d,
        body_vel,
        body_ang_vel
    ])
    
    if add_time:
        # Add additional dimensions for future poses (75 dims)
        # This includes time and any other features needed for the transformer
        # For now, we'll just pad with zeros except for normalized time
        time_features = torch.zeros(75)  # Additional features for future poses
        time_features[0] = frame_idx / len(motion.rotation)  # Normalized time
        obs = torch.cat([obs, time_features])
    
    return obs

def load_motion_data(motion_file: str):
    """Load motion data from .npy file"""
    motion_data = np.load(motion_file, allow_pickle=True).item()
    print("Motion data structure:")
    # for k, v in motion_data.items():
    #     if isinstance(v, np.ndarray):
    #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    #     else:
    #         print(f"  {k}: {type(v)}")
    #         if k == 'skeleton_tree':
    #             print("\nSkeleton tree structure:")
    #             for joint_name, joint_info in v.items():
    #                 print(f"  {joint_name}:")
    #                 for info_key, info_val in joint_info.items():
    #                     print(f"    {info_key}: {info_val}")
    
    # Create SkeletonTree
    skeleton_tree = SkeletonTree(
        node_names=motion_data['skeleton_tree']['node_names'],
        parent_indices=torch.from_numpy(motion_data['skeleton_tree']['parent_indices']['arr']).int(),
        local_translation=torch.from_numpy(motion_data['skeleton_tree']['local_translation']['arr']).float()
    )
    
    # Create SkeletonMotion
    motion = SkeletonMotion.from_dict(motion_data)
    
    print(f"\nNumber of frames: {len(motion.rotation)}")
    return motion

def get_motion_slice(motion: SkeletonMotion, start_idx: int, num_future_steps: int = 15):
    """
    Get a slice of motion data starting at start_idx
    Returns current pose and future poses
    """
    # Get current pose (no time dimension needed)
    current_pose = construct_pose_obs(motion, start_idx, add_time=False)
    
    # Get future poses (with time dimension)
    future_poses = []
    for i in range(num_future_steps):
        future_idx = min(start_idx + i + 1, len(motion.rotation) - 1)
        future_pose = construct_pose_obs(motion, future_idx, add_time=True)  # Add time dimension
        future_poses.append(future_pose)
    future_poses = torch.stack(future_poses)
    
    return current_pose, future_poses

def load_motion_tracker(checkpoint_path: str, device: str = "cuda"):
    """
    Load the motion tracker model from checkpoint
    """
    # Load config and checkpoint
    config = OmegaConf.load(Path(checkpoint_path).parent / "config.yaml")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = PPOModel(config.agent.config.model.config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    return model

def prepare_input(current_pose, future_poses, heightmap, device="cuda"):
    """
    Prepare input dictionary for the model
    Args:
        current_pose: Current pose tensor [batch_size, self_obs_size]
        future_poses: Future poses tensor [batch_size, num_future_steps, num_obs_per_target_pose]
        heightmap: Heightmap tensor [batch_size, num_samples]
    """
    # Ensure inputs are on the correct device and type
    current_pose = current_pose.to(device).float()
    future_poses = future_poses.to(device).float()
    heightmap = heightmap.to(device).float()
    
    # Ensure correct shapes
    batch_size = current_pose.shape[0]
    if len(current_pose.shape) == 1:
        current_pose = current_pose.unsqueeze(0)
    if len(future_poses.shape) == 2:
        future_poses = future_poses.unsqueeze(0)
    if len(heightmap.shape) == 1:
        heightmap = heightmap.unsqueeze(0)
        
    return {
        "self_obs": current_pose,  # [batch_size, 358]
        "mimic_target_poses": future_poses,  # [batch_size, 15, 433]
        "terrain": heightmap  # [batch_size, 256]
    }

def get_next_action(model, current_pose, future_poses, heightmap, device="cuda"):
    """
    Get next action from the model
    """
    with torch.no_grad():
        inputs = prepare_input(current_pose, future_poses, heightmap, device)
        action = model.act(inputs, mean=True)  # Use mean=True for deterministic actions
    return action

def visualize_pose(motion: SkeletonMotion, frame_idx: int, action=None, ax=None):
    """
    Visualize a single pose frame with optional angular velocity vectors
    
    Args:
        motion: SkeletonMotion object
        frame_idx: Frame index to visualize
        action: Optional action vector [69] showing angular velocity targets (excluding root)
        ax: Optional matplotlib axis
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get current joint positions
    positions = motion.global_translation[frame_idx]  # [24, 3]
    
    # Plot current pose
    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o', s=100, label='Current Pose')
    
    # Draw lines between joints based on parent relationships
    for joint_idx in range(1, len(positions)):  # Skip root
        parent_idx = motion.skeleton_tree.parent_indices[joint_idx]
        if parent_idx >= 0:  # If joint has a parent
            start_pos = positions[parent_idx]
            end_pos = positions[joint_idx]
            ax.plot([start_pos[0], end_pos[0]], 
                   [start_pos[1], end_pos[1]], 
                   [start_pos[2], end_pos[2]], 'r-', linewidth=2)
    
    # If action is provided, show angular velocity vectors
    if action is not None:
        # Reshape action into [23, 3] for angular velocities (excluding root)
        angular_vels = action.reshape(-1, 3)
        
        # Scale factor for visualization (adjust based on magnitude)
        scale = 0.3
        
        # Draw angular velocity vectors for each joint
        for joint_idx in range(1, len(positions)):  # Skip root
            joint_pos = positions[joint_idx]
            ang_vel = angular_vels[joint_idx-1] * scale
            
            # Only draw arrows if angular velocity magnitude is significant
            if torch.norm(ang_vel) > 0.01:
                # Draw arrow showing rotation axis and magnitude
                ax.quiver(joint_pos[0], joint_pos[1], joint_pos[2],
                         ang_vel[0], ang_vel[1], ang_vel[2],
                         color='g', alpha=0.7, arrow_length_ratio=0.2,
                         label='Angular Velocity' if joint_idx == 1 else None)
                
                # Optionally, draw a small arc to indicate rotation
                # (this is approximate and just for visualization)
                theta = torch.norm(ang_vel)
                if theta > 0:
                    # Create points for an arc
                    t = torch.linspace(0, theta, 20)
                    radius = 0.05
                    axis = ang_vel / theta
                    
                    # Create a basis for the rotation plane
                    if abs(axis[2]) < 0.9:
                        basis_x = torch.tensor([0., 0., 1.])
                    else:
                        basis_x = torch.tensor([1., 0., 0.])
                    basis_y = torch.cross(axis, basis_x)
                    basis_y = basis_y / torch.norm(basis_y)
                    basis_x = torch.cross(basis_y, axis)
                    
                    # Create arc points
                    arc_points = joint_pos + radius * (torch.cos(t)[:, None] * basis_x + 
                                                     torch.sin(t)[:, None] * basis_y)
                    
                    # Plot arc
                    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                           'g:', alpha=0.5, linewidth=1)
    
    # Set equal aspect ratio and labels
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    ax.set_title(f'Frame {frame_idx}', pad=20)
    
    # Add legend
    if action is not None:
        ax.legend()
    
    # Set consistent view limits
    positions_min = positions.min(dim=0)[0]
    positions_max = positions.max(dim=0)[0]
    center = (positions_max + positions_min) / 2
    max_range = (positions_max - positions_min).max().item()
    
    # Add some padding to the view limits
    padding = max_range * 0.3
    ax.set_xlim(center[0] - max_range/2 - padding, center[0] + max_range/2 + padding)
    ax.set_ylim(center[1] - max_range/2 - padding, center[1] + max_range/2 + padding)
    ax.set_zlim(center[2] - max_range/2 - padding, center[2] + max_range/2 + padding)
    
    # Set a good viewing angle
    ax.view_init(elev=15, azim=45)
    
    # Add a ground plane
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 2),
                        np.linspace(min_y, max_y, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    # Customize grid and background
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    return ax

def create_animation(motion: SkeletonMotion, start_frame: int = 0, num_frames: int = 100):
    """Create an animation of the motion sequence"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        visualize_pose(motion, start_frame + frame, ax)
        return ax,
    
    num_frames = min(num_frames, len(motion.rotation) - start_frame)
    anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                 interval=50, blit=True)
    return anim

# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = load_motion_tracker("data/pretrained_models/motion_tracker/smpl/last.ckpt", device)
    
    # Load motion data
    motion = load_motion_data("data/motions/smpl_humanoid_walk.npy")
    
    # Create flat heightmap (no terrain features)
    batch_size = 1
    heightmap = torch.zeros(batch_size, 256).to(device)  # [batch_size, terrain_obs_num_samples]
    
    # Process frames and collect actions
    actions = []
    frame_indices = []
    for start_idx in range(0, min(100, len(motion.rotation)), 10):
        print(f"\nProcessing frame {start_idx}")
        
        # Get motion slice
        current_pose, future_poses = get_motion_slice(motion, start_idx)
        
        # Print input shapes for debugging
        print(f"Current pose shape: {current_pose.shape}")
        print(f"Future poses shape: {future_poses.shape}")
        print(f"Current pose range: [{current_pose.min():.3f}, {current_pose.max():.3f}]")
        
        # Get action
        action = get_next_action(model, current_pose, future_poses, heightmap)
        print(f"Action shape: {action.shape}")
        print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
        actions.append(action)
        frame_indices.append(start_idx)
    
    # Create a grid of frames with actions
    print("\nCreating frame grid visualization with actions...")
    fig = plt.figure(figsize=(20, 10))
    
    # Select 6 frames evenly spaced through the processed frames
    selected_indices = np.linspace(0, len(frame_indices)-1, 6, dtype=int)
    
    for i, idx in enumerate(selected_indices, 1):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        frame_idx = frame_indices[idx]
        action = actions[idx].cpu().squeeze()  # Get corresponding action
        visualize_pose(motion, frame_idx, action, ax)
        ax.set_title(f'Frame {frame_idx}\nAction Magnitude: {torch.norm(action):.2f}')
    
    plt.tight_layout()
    plt.savefig('motion_frames_with_actions_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved frame grid with actions as 'motion_frames_with_actions_grid.png'")
    
    # Create animation with actions
    print("\nCreating animation with actions...")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        # Find closest processed frame and its action
        closest_idx = min(range(len(frame_indices)), 
                         key=lambda i: abs(frame_indices[i] - frame))
        action = actions[closest_idx].cpu().squeeze()
        visualize_pose(motion, frame, action, ax)
        return ax,
    
    # Create animation with more frames and smoother playback
    num_frames = min(200, len(motion.rotation))
    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                 interval=33,  # ~30 fps
                                 blit=False)
    
    # Save animation with higher quality settings
    print("Saving animation...")
    anim.save('motion_visualization_with_actions.mp4', 
              writer='ffmpeg',
              fps=30,
              dpi=150,
              bitrate=2000)
    print("Animation saved as 'motion_visualization_with_actions.mp4'")