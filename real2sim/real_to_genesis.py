from pathlib import Path
from optitrack_vendor.NatNetClient import setup_optitrack
from genesis_viewer import GenesisViewer
import threading
import argparse
import time

from gs_env.real.low_state_handler import LowStateMsgHandler
from gs_env.sim.envs.config.registry import EnvArgsRegistry


def main(args):
    client = setup_optitrack(
        server_address=args.server_ip,
        client_address=args.client_ip,
        use_multicast=args.use_multicast,
    )

    # start a thread to client.run()
    thread = threading.Thread(target=client.run)
    thread.start()

    if not client:
        print("Failed to setup OptiTrack client")
        exit(1)

    frame = client.get_frame() # eventually this will stuck, so we put it before env init

    offset_config_path = Path(__file__).resolve().parent / "config" / (args.offset_config + ".yaml")
    genesis_env = GenesisViewer(visualize=True, offset_config_path=offset_config_path)

    from config.camera_config import Camera_Calibrations, World_Rotation
    genesis_env.initialize_cameras(Camera_Calibrations, World_Rotation)

    if args.run == "cb1" or args.run == "cb2":
        genesis_env.Real2Sim_offset_setup(args)
    else:
        genesis_env.Real2Sim_setup(args)
    
    cfg = EnvArgsRegistry["real2sim_default"]
    low_state_handler = LowStateMsgHandler(cfg)
    low_state_handler.init()

    tic = 0
    while True:
        tic += 1
        frame = client.get_frame()
        frame_number = client.get_frame_number()

        if args.run == "cb1" or args.run == "cb2":
            genesis_env.Real2Sim_offset_step(frame)
            if args.run == "cb1":
                genesis_env.calibrate_by_world(frame)
            elif args.run == "cb2":
                joint_qpos = low_state_handler.joint_pos
                genesis_env.calibrate_by_fixed_link(frame, joint_qpos)
            if tic % 100 == 0:
                new_offset_config_path = Path(__file__).resolve().parent / "config" / (args.offset_config + '_' + args.run + ".yaml")
                genesis_env.save_offsets(new_offset_config_path)
        else:
            genesis_env.Real2Sim_step(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.0.232")
    parser.add_argument("--client_ip", type=str, default="192.168.0.128")
    parser.add_argument("--use_multicast", type=bool, default=False)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--run", type=str, choices=["cb1", "cb2", "track"], default="track")
    parser.add_argument("--offset_config", type=str, default="offset_default")
    args = parser.parse_args()
    main(args)
    