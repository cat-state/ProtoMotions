# pyright: basic
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
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree, SkeletonState
import xml.etree.ElementTree as ET
import sys
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.envs.base_env.env_utils.humanoid_utils import (
    compute_humanoid_observations,
    build_pd_action_offset_scale,
    dof_to_obs
)
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


def quat_to_matrix(quat: torch.Tensor):
    """Convert quaternion to rotation matrix"""
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1)
    
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

from protomotions.utils.motion_lib import MotionLib
def _compute_dof_offsets(dof_names: list[str]) -> list[int]:
    """
    Compute and return offsets where consecutive bodies' DOFs start.

    Args:
        dof_names (List[str]): List of DOF names.

    Returns:
        List[int]: A list of offsets indicating the start of each new set of DOFs.
    """
    dof_offsets: list[int] = []
    previous_dof_name: str = "null"
    for dof_offset, dof_name in enumerate(dof_names):
        if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
            previous_dof_name = dof_name[:-2]
            dof_offsets.append(dof_offset)
    dof_offsets.append(len(dof_names))
    return dof_offsets



def load_motion_data(motion_file: str, robot_cfg) -> MotionLib:
    key_body_ids = [
        robot_cfg.body_names.index(key_body_name)
        for key_body_name in robot_cfg.key_bodies
    ]
    dof_offsets = []

    return MotionLib(
        motion_file=motion_file,
        dof_body_ids=range(1, 24),
        dof_offsets=_compute_dof_offsets(robot_cfg.dof_names),
        key_body_ids=key_body_ids,
        ref_height_adjust=0.0,
        fix_motion_heights=True
    )

from protomotions.envs.base_env.env_utils.humanoid_utils import compute_humanoid_observations_max

from protomotions.envs.mimic.mimic_utils import build_max_coords_target_poses_future_rel

def load_motion_tracker(checkpoint_path: str, device: str = "cuda"):
    """
    Load the motion tracker model from checkpoint
    """
    # Load config and checkpoint
    config = OmegaConf.load(Path(checkpoint_path).parent / "config.yaml")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Create model
    model = PPOModel(config.agent.config.model.config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    return model, config

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
        "self_obs": current_pose,  # [batch_size, obs_size]
        "mimic_target_poses": future_poses,  # [batch_size, num_future_steps, obs_size_with_time]
        "terrain": heightmap  # [batch_size, num_samples]
    }

def get_next_action(model, current_pose, future_poses, heightmap, device="cuda"):
    """
    Get next action from the model
    """
    with torch.no_grad():
        inputs = prepare_input(current_pose, future_poses, heightmap, device)
        action = model.act(inputs, mean=True)  # Use mean=True for deterministic actions
    return action

def parse_joint_info(xml_file):
    """Parse joint axes and limits from the SMPL humanoid XML file"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot().find("worldbody")
        
        joint_info = {}
        joint_order = []  # To maintain order of joints as they appear
        
        # Parse all joints
        for joint in root.findall(".//joint"):
            name = joint.get('name')
            axis_str = joint.get('axis')
            range_str = joint.get('range')
            
            if not all([name, axis_str, range_str]):
                print(f"Warning: Joint {name} missing required attributes")
                continue
                
            try:
                axis = [float(x) for x in axis_str.split()]
                angle_range = [float(x) for x in range_str.split()]
                
                joint_info[name] = {
                    'axis': torch.tensor(axis),
                    'range': torch.tensor(angle_range),
                    'range_radians': torch.tensor([np.deg2rad(angle_range[0]), np.deg2rad(angle_range[1])])
                }
                joint_order.append(name)
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse joint {name}: {e}")
                continue
        
        if not joint_info:
            raise ValueError("No valid joints found in XML file")
            
        return joint_info, joint_order
        
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return None, None

def remap_action_to_angle(action_value, joint_range):
    """Remap action value from [-1, 1] to joint angle range in radians"""
    min_angle, max_angle = joint_range[0], joint_range[1]
    return min_angle + (action_value + 1.0) * 0.5 * (max_angle - min_angle)

from isaac_utils.isaac_utils.rotations import quat_from_euler_xyz, quat_mul_norm, quat_from_angle_axis, quat_mul
from torch import Tensor
from typing import List
from isaac_utils import torch_utils, rotations

@torch.jit.script
def dof_to_local(pose: Tensor, dof_offsets: List[int], w_last: bool) -> Tensor:
    """Convert degrees of freedom (DoF) representation to local rotations.

    Args:
        pose: Input pose tensor with shape [..., total_dofs]
        dof_offsets: List of DoF offsets for each joint
        w_last: Whether quaternion w component is last

    Returns:
        Local rotation quaternions with shape [..., num_joints, 4]
    """
    num_joints = len(dof_offsets) - 1
    assert pose.shape[-1] == dof_offsets[-1], "Pose size must match total DoFs"

    # Initialize output tensor for local rotations
    local_rot_shape = pose.shape[:-1] + (num_joints, 4)
    local_rot = torch.zeros(local_rot_shape, device=pose.device)

    # Convert each joint's DoFs to quaternion
    for joint_idx in range(num_joints):
        start_dof = dof_offsets[joint_idx]
        end_dof = dof_offsets[joint_idx + 1]
        dof_size = end_dof - start_dof
        joint_pose = pose[..., start_dof:end_dof]

        if dof_size == 3:  # Spherical joint (3 DoF)
            joint_quat = torch_utils.exp_map_to_quat(joint_pose, w_last)
        elif dof_size == 1:  # Revolute joint (1 DoF)
            if (joint_idx % 3) == 0:
                axis = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=joint_pose.dtype, device=pose.device
                )
            elif (joint_idx % 3) == 1:
                axis = torch.tensor(
                    [0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device
                )
            elif (joint_idx % 3) == 2:
                axis = torch.tensor(
                    [0.0, 0.0, 1.0], dtype=joint_pose.dtype, device=pose.device
                )
            else:
                assert False, "you broke math"
            joint_quat = rotations.quat_from_angle_axis(
                joint_pose[..., 0], axis, w_last
            )
        else:
            raise ValueError(f"Unsupported joint type with {dof_size} DoF")

        local_rot[..., joint_idx, :] = joint_quat

    return local_rot


def visualize_pose(motion: SkeletonMotion, frame_idx: int, action=None, ax=None, joint_info=None, joint_order=None):
    """
    Visualize a single pose frame with optional target positions
    
    Args:
        motion: SkeletonMotion object
        frame_idx: Frame index to visualize
        action: Optional action vector showing target joint angles (excluding root)
        ax: Optional matplotlib axis
        joint_info: Dictionary containing joint information
        joint_order: List of joint names in the order they appear
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
    
    # If action is provided, show target positions
    if action is not None and joint_info is not None and joint_order is not None:
        # Get current local rotations and root translation
        current_local_rot = motion.local_rotation[frame_idx]
        current_root_trans = motion.root_translation[frame_idx]
        
        # Apply PD action offset and scale to get target angles
        dof_offsets = [i*3 for i in range(24+1)]  # +1 for the end offset
        
        # Get DOF limits from joint info
        dof_limits_lower = torch.zeros(69, device=action.device)
        dof_limits_upper = torch.zeros(69, device=action.device)
        
        joint_idx = 0
        for name, info in joint_info.items():
            dof_limits_lower[joint_idx] = info['range_radians'][0]
            dof_limits_upper[joint_idx] = info['range_radians'][1]
            joint_idx += 1

        pd_offset, pd_scale = build_pd_action_offset_scale(
            dof_offsets[:-1], dof_limits_lower, dof_limits_upper, action.device
        )
        
        # Convert action to target angles
        pd_target = pd_offset + pd_scale * action
        
        # Convert target angles to local rotations
        target_local_rot = dof_to_local(pd_target, dof_offsets[:-1], w_last=True)
        
        # Create target pose state directly using SkeletonState
        target_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=motion.skeleton_tree,
            r=torch.cat([current_local_rot[0:1], target_local_rot], dim=0),
            t=current_root_trans,
            is_local=True
        )
        
        # Get target global positions
        target_positions = target_state.global_translation  # [24, 3]
        
        # Draw target positions
        ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
                  c='g', marker='o', s=50, alpha=0.5, label='Target Position')
        
        # Draw lines from current to target positions
        for i in range(1, len(positions)):  # Skip root
            ax.plot([positions[i, 0], target_positions[i, 0]],
                   [positions[i, 1], target_positions[i, 1]],
                   [positions[i, 2], target_positions[i, 2]],
                   'g--', alpha=0.5, linewidth=1)
    
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

def quat_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    
    return torch.tensor([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def create_animation(motion: MotionLib, start_frame: int = 0, num_frames: int = 100):
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
    torch.set_grad_enabled(False)
    # Parse joint info at startup
    xml_path = "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    joint_info, joint_order = parse_joint_info(xml_path)
    if joint_info is None:
        print("Failed to parse joint info from XML file. Exiting.")
        sys.exit(1)
    
    print(f"Successfully parsed {len(joint_info)} joints from XML")
    
    # Load model
    model, motion_tracker_cfg = load_motion_tracker("data/pretrained_models/motion_tracker/smpl/last.ckpt", device)
    robot_cfg = motion_tracker_cfg.robot
    motion = load_motion_data("data/motions/smpl_humanoid_walk.npy", robot_cfg)
    
    # Create flat heightmap (no terrain features)
    batch_size = 1
    heightmap = torch.zeros(batch_size, 256).to(device)  # [batch_size, terrain_obs_num_samples]
    
    # Process frames and collect actions
    actions = []
    frame_indices = []
    print(motion.motion_ids)
    print(motion)
    dt = 1.0/30.0
    length = motion.get_motion_length(torch.tensor([0]))
    num_frames = int(torch.floor(length / dt).item())
    for start_idx in range(0, num_frames, 1):
        print(f"\nProcessing frame {start_idx}")
        
        time_offsets = torch.arange(1, 15 + 1) * dt
        robot_state = motion.get_motion_state(
            torch.tensor([0 for _ in range(16)]), 
            torch.tensor([dt * (start_idx + i) for i in range(16)])
        )
        assert robot_state.rigid_body_pos is not None 
        assert robot_state.rigid_body_rot is not None   
        assert robot_state.rigid_body_vel is not None
        assert robot_state.rigid_body_ang_vel is not None
        self_obs = compute_humanoid_observations_max(
            body_pos=robot_state.rigid_body_pos,
            body_rot=robot_state.rigid_body_rot,
            body_vel=robot_state.rigid_body_vel,
            body_ang_vel=robot_state.rigid_body_ang_vel,
            ground_height=torch.zeros(1, 1),
            local_root_obs=True,
            root_height_obs=True,
            w_last=True
        )
        target_obs = build_max_coords_target_poses_future_rel(
            cur_gt=robot_state.rigid_body_pos[:1],
            cur_gr=robot_state.rigid_body_rot[:1],
            flat_target_pos=robot_state.rigid_body_pos[1:],
            flat_target_rot=robot_state.rigid_body_rot[1:],
            num_envs=1,
            num_future_steps=15,
            w_last=True
        ).cuda()
        target_obs = target_obs.reshape(1, 15, -1)
        target_obs = torch.cat([target_obs, time_offsets[None, :, None].cuda()], dim=-1)

        inputs = {
            "self_obs": self_obs.cuda()[:1],  # [batch_size, obs_size]
            "mimic_target_poses": target_obs,  # [batch_size, num_future_steps, obs_size_with_time]
            "terrain": heightmap  # [batch_size, num_samples]
        }
        action = model.act(inputs, mean=True)
        print(f"Action shape: {action.shape}")
        print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
        actions.append(action)
        frame_indices.append(start_idx)
    
    # Create a grid of frames with actions
    print("\nCreating frame grid visualization with actions...")
    fig = plt.figure(figsize=(20, 15))  # Made figure taller to accommodate action plot
    
    # Select 6 frames evenly spaced through the processed frames
    selected_indices = np.linspace(0, len(frame_indices)-1, 6, dtype=int)
    
    # Create subplot grid: 3 rows (2 for poses, 1 for action plot) x 3 columns
    gs = plt.GridSpec(3, 3)
    
    for i, idx in enumerate(selected_indices, 1):
        row = (i-1) // 3
        col = (i-1) % 3
        ax = fig.add_subplot(gs[row, col], projection='3d')
        frame_idx = frame_indices[idx]
        action = actions[idx].cpu().squeeze()  # Get corresponding action
        visualize_pose(motion.state.motions[0], frame_idx, action, ax, joint_info, joint_order)  # Pass joint_info and joint_order
        ax.set_title(f'Frame {frame_idx}\nAction Magnitude: {torch.norm(action):.2f}')
    
    # Add action trajectory plot in bottom row
    ax_actions = fig.add_subplot(gs[2, :])
    actions_array = torch.stack(actions).cpu().numpy()
    
    # Plot a subset of joints for clarity
    joint_indices = [
        (0, 3),    # Left Hip
        (12, 15),  # Right Hip
        (24, 27),  # Torso
        (39, 42),  # Head
        (42, 45),  # Left Shoulder
        (60, 63),  # Right Shoulder
    ]  # Example joint indices (adjust based on your needs)
    
    joint_names = [
        "Left Hip", "Right Hip", "Torso",
        "Head", "Left Shoulder", "Right Shoulder"
    ]
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    # Get current angles for comparison
    current_angles = []
    for frame_idx in frame_indices:
        # Extract Euler angles from current rotations
        euler_angles = []
        for joint_idx in range(1, 24):  # Skip root
            rot_quat = motion.state.motions[0].local_rotation[frame_idx, joint_idx]

            # Convert quaternion to axis-angle
            angle = 2 * torch.acos(rot_quat[3])  # w component
            if angle > 0.01:
                axis = rot_quat[:3] / torch.sin(angle/2)
                euler_angles.extend(axis * angle)
            else:
                euler_angles.extend([0, 0, 0])
        current_angles.append(euler_angles)
    current_angles = np.array(current_angles)
    
   
    # Create default joint limits
    dof_limits_lower = torch.ones(69, device=actions[0].device)
    dof_limits_upper = torch.ones(69, device=actions[0].device) 
    # Try to use joint info if available

    dof_offsets = [i*3 for i in range(24+1)]  # +1 for the end offset

    joint_idx = 0
    for name, info in joint_info.items():
        dof_limits_lower[joint_idx] = info['range_radians'][0]
        dof_limits_upper[joint_idx] = info['range_radians'][1]
        joint_idx += 1

    pd_offset, pd_scale = build_pd_action_offset_scale(
        dof_offsets[:-1], dof_limits_lower, dof_limits_upper, action.device
    )

    # Plot both target and current angles
    for (start_idx, end_idx), name, color in zip(joint_indices, joint_names, colors):
        for j in range(start_idx, end_idx):
            # Convert action to target angles using PD scale
            print(f"{actions_array.shape=}")
            pd_targets = pd_offset[j] + pd_scale[j] * actions_array[:, 0, j]
            
            # Plot target angles
            ax_actions.plot(frame_indices, pd_targets, 
                          alpha=0.7, color=color, linestyle='-',
                          label=f'{name} Target (axis {j-start_idx})' if j == start_idx else None)
            # Plot current angles
            ax_actions.plot(frame_indices, current_angles[:, j], 
                          alpha=0.3, color=color, linestyle='--',
                          label=f'{name} Current (axis {j-start_idx})' if j == start_idx else None)
    
    ax_actions.set_xlabel('Frame')
    ax_actions.set_ylabel('Angle (radians)')
    ax_actions.set_title('Joint Angles Over Time (Solid: Target, Dashed: Current)')
    ax_actions.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_actions.grid(True)
    
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
        visualize_pose(motion.state.motions[0], frame, action, ax, joint_info, joint_order)  # Pass joint_info and joint_order
        return ax,
    
    # Create animation with more frames and smoother playback
    num_frames = min(1000, len(motion.state.motions[0].rotation))
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
