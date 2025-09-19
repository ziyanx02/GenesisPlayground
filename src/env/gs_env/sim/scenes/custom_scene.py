from typing import Any

import genesis as gs
import torch

from gs_env.common.bases.base_scene import BaseSimScene
from gs_env.sim.scenes.config.schema import CustomSceneArgs


class CustomScene(BaseSimScene):
    def __init__(
        self,
        num_envs: int,
        args: CustomSceneArgs,
        show_viewer: bool = False,
        show_fps: bool = False,
        n_envs_per_row: int | None = None,
        env_spacing: tuple[float, float] = (1.0, 1.0),
        img_resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
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
        for object in args.objects:
            obj_type: str = object.get("type", "")
            if obj_type.lower() in ["obj", "stl", "ply"]:
                self._scene.add_entity(
                    gs.morphs.Mesh(
                        file=object["path"],
                        scale=object.get("scale", 1.0),
                        pos=object["position"],
                        euler=object["orientation"],
                        fixed=object.get("fixed", True),
                        visualization=object.get("visualization", True),
                        collision=object.get("collision", True),
                    )
                )
            elif obj_type.lower() == "urdf":
                self._scene.add_entity(
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
                self._scene.add_entity(
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
                self._scene.add_entity(
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
                self._scene.add_entity(
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

        #
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._n_envs_per_row = n_envs_per_row
        self._center_envs_at_origin = args.center_envs_at_origin
        self._compile_kernels = args.compile_kernels

    def reset(self, envs_idx: torch.IntTensor) -> None:
        self._scene.reset(envs_idx=envs_idx)

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
