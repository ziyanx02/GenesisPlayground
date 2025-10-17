from typing import Any

import genesis as gs
import torch

from gs_env.common.bases.base_scene import BaseSimScene
from gs_env.sim.scenes.config.schema import CustomSceneArgs


class CustomScene(BaseSimScene):
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        args: CustomSceneArgs,
        show_viewer: bool = False,
        show_fps: bool = False,
        n_envs_per_row: int | None = None,
        env_spacing: tuple[float, float] = (1.0, 1.0),
        img_resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self._device = device
        #
        # _renderer = (
        #     gs.options.renderers.BatchRenderer(
        #         use_rasterizer=False,
        #     )
        # )
        self._scene = gs.Scene(
            sim_options=args.sim_options,
            tool_options=args.tool_options,
            rigid_options=args.rigid_options,
            avatar_options=args.avatar_options,
            mpm_options=args.mpm_options,
            fem_options=args.fem_options,
            sf_options=args.sf_options,
            vis_options=args.vis_options,
            viewer_options=args.viewer_options,
            show_FPS=show_fps,
            show_viewer=show_viewer,
            # renderer=_renderer,
        )
        #
        if not args.remove_ground:
            self._plane = self._scene.add_entity(
                gs.morphs.Plane(normal=args.normal),
            )
        self._objects = {}
        for object in args.objects:
            obj_type: str = object.get("type", "")
            if obj_type.lower() in ["obj", "stl", "ply"]:
                obj = self._scene.add_entity(
                    gs.morphs.Mesh(
                        file=object["path"],
                        scale=object.get("scale", 1.0),
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    ),
                    surface=gs.surfaces.Plastic(color=object.get("color", (0.5, 0.5, 0.5))),
                )
            elif obj_type.lower() == "urdf":
                obj = self._scene.add_entity(
                    gs.morphs.URDF(
                        file=object["path"],
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            elif obj_type.lower() == "box":
                obj = self._scene.add_entity(
                    gs.morphs.Box(
                        size=object["size"],
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            elif obj_type.lower() == "cylinder":
                obj = self._scene.add_entity(
                    gs.morphs.Cylinder(
                        radius=object["radius"],
                        height=object["height"],
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            elif obj_type.lower() == "sphere":
                obj = self._scene.add_entity(
                    gs.morphs.Sphere(
                        radius=object["radius"],
                        pos=object["position"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            else:
                raise ValueError(f"Unsupported object type: {obj_type}")
            self._objects[object["name"]] = obj
            print(f"Added object: {object['name']}")

        #
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._n_envs_per_row = n_envs_per_row
        self._center_envs_at_origin = args.center_envs_at_origin
        self._compile_kernels = args.compile_kernels

    def reset(self, envs_idx: torch.IntTensor) -> None:
        self._scene.reset(envs_idx=envs_idx)

    def set_obj_pose(
        self,
        name: str,
        envs_idx: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        quat: torch.Tensor | None = None,
    ) -> None:
        assert name in self._objects, f"Object {name} not found in scene"
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self._device)
        if pos is not None:
            assert pos.shape == (len(envs_idx), 3), (
                "Position must be a tensor of shape (num_envs, 3)"
            )
        if quat is not None:
            assert quat.shape == (len(envs_idx), 4), (
                "Quaternion must be a tensor of shape (num_envs, 4)"
            )
        obj = self._objects[name]
        if pos is not None:
            obj.set_pos(pos, envs_idx=envs_idx)
        if quat is not None:
            obj.set_quat(quat, envs_idx=envs_idx)

    def __getattr__(self, item: str) -> Any:
        if hasattr(self._scene, item):
            return getattr(self._scene, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    @property
    def scene(self) -> gs.Scene:
        """Returns the underlying genesis scene."""
        return self._scene

    @property
    def env_spacing(self) -> tuple[float, float]:
        """Returns the spacing between environments."""
        return self._env_spacing

    @property
    def n_envs_per_row(self) -> int | None:
        return self._n_envs_per_row

    @property
    def center_envs_at_origin(self) -> bool:
        return self._center_envs_at_origin

    @property
    def num_envs(self) -> int:
        return self._num_envs
