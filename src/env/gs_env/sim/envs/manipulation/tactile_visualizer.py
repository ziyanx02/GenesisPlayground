"""Tactile sensor visualization for WUJI hand in-hand rotation."""

from typing import Any

import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch


# YZ offsets for tactile visualization (from Genesis tactile_field_hand.py)
LINK_YZ_OFFSETS = {
    "palm_link": (0.0, 0.0),
    "finger1_link1": (0.05, 0.02),
    "finger1_link2": (0.05, 0.05),
    "finger1_link3": (0.05, 0.08),
    "finger1_link4": (0.05, 0.11),
    "finger1_tip_link": (0.05, 0.14),
    "finger2_link1": (0.02, 0.06),
    "finger2_link2": (0.02, 0.10),
    "finger2_link3": (0.02, 0.14),
    "finger2_link4": (0.02, 0.18),
    "finger2_tip_link": (0.02, 0.21),
    "finger3_link1": (0.0, 0.06),
    "finger3_link2": (0.0, 0.10),
    "finger3_link3": (0.0, 0.14),
    "finger3_link4": (0.0, 0.18),
    "finger3_tip_link": (0.0, 0.21),
    "finger4_link1": (-0.02, 0.06),
    "finger4_link2": (-0.02, 0.10),
    "finger4_link3": (-0.02, 0.14),
    "finger4_link4": (-0.02, 0.18),
    "finger4_tip_link": (-0.02, 0.21),
    "finger5_link1": (-0.04, 0.04),
    "finger5_link2": (-0.04, 0.08),
    "finger5_link3": (-0.04, 0.12),
    "finger5_link4": (-0.04, 0.16),
    "finger5_tip_link": (-0.04, 0.19),
}


class TactileVisualizer:
    """Real-time tactile force visualization for WUJI hand."""

    def __init__(
        self,
        tactile_sensors: dict[str, Any],
        tactile_points: dict[str, np.ndarray],
        tactile_sensor_configs: dict[str, dict[str, int]],
    ):
        """
        Initialize tactile visualizer.

        Args:
            tactile_sensors: Dictionary of sensor objects {link_name: sensor}
            tactile_points: Dictionary of tactile points {link_name: np.array of positions}
            tactile_sensor_configs: Dictionary of sensor configs {link_name: {'num_points': int, ...}}
        """
        self.tactile_sensors = tactile_sensors
        self.tactile_points = tactile_points
        self.tactile_sensor_configs = tactile_sensor_configs
        self.sensor_link_names = sorted(tactile_sensors.keys())

        # Setup matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))

        # Make sure the window is shown
        self.fig.show()

        self.ax.set_title("Full Hand Tactile Force Field")
        self.ax.set_xlabel("Y (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.set_aspect('equal')

        # Create scatter plots for each sensor link
        self.scatter_plots = {}
        for link_name in self.sensor_link_names:
            local_positions = self.tactile_points[link_name]

            # Get YZ offset for this link
            yz_offset = LINK_YZ_OFFSETS.get(link_name, (0.0, 0.0))

            # Apply offset to local positions (project to YZ plane and offset)
            if link_name != "finger1_link2":
                # Most links: use Y, Z coordinates
                offset_positions = local_positions[:, 1:3].copy()  # Y, Z coordinates
                offset_positions[:, 0] += yz_offset[0]  # Y offset
                offset_positions[:, 1] += yz_offset[1]  # Z offset
            else:
                # finger1_link2: project to XZ plane
                offset_positions = local_positions[:, [0, 2]].copy()  # X, Z coordinates
                offset_positions[:, 0] += yz_offset[0]  # X offset
                offset_positions[:, 1] += yz_offset[1]  # Z offset

            # Create scatter plot for tactile points
            scatter = self.ax.scatter(
                offset_positions[:, 0],  # Y coordinate (with offset)
                offset_positions[:, 1],  # Z coordinate (with offset)
                c=np.zeros(len(local_positions)),
                cmap='hot',
                vmin=0,
                vmax=10,
                s=20,
                edgecolors='black',
                linewidths=0.5,
                label=link_name
            )
            self.scatter_plots[link_name] = (scatter, offset_positions)

        # Add colorbar
        self.cbar = plt.colorbar(scatter, ax=self.ax, label='Force (N)')
        plt.tight_layout()

        # Force the window to show
        plt.show(block=False)
        plt.pause(0.1)  # Give it time to render

    def update(self) -> None:
        """Update the visualization with current sensor readings."""
        max_force_global = 0.0

        for link_name in self.sensor_link_names:
            sensor = self.tactile_sensors[link_name]
            config = self.tactile_sensor_configs[link_name]

            # Read sensor data (returns num_points * 3 forces)
            force_field_full = sensor.read()
            num_points = config['num_points']

            # Reshape to (num_points, 3)
            force_field_3d = force_field_full.reshape(num_points, 3)

            # Compute force magnitudes (using all 3 components for visualization)
            force_magnitudes = torch.norm(force_field_3d, dim=-1)  # (num_points,)

            # Update scatter plot colors
            force_mag_np = force_magnitudes.cpu().numpy()
            scatter, offset_positions = self.scatter_plots[link_name]
            scatter.set_array(force_mag_np)

            # Track max force for global scaling
            max_force_global = max(max_force_global, force_mag_np.max())

        # Auto-scale colorbar globally
        vmax = max(max_force_global, 1.0)
        for link_name in self.sensor_link_names:
            scatter, _ = self.scatter_plots[link_name]
            scatter.set_clim(vmin=0, vmax=vmax)

        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        """Close the visualization."""
        plt.ioff()
        plt.close(self.fig)
