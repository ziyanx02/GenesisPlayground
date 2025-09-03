import gs_env.common.bases.base_sensor as base_sensor
from gs_env.common.bases.base_scene import BaseSimScene


class Camera(base_sensor.BaseCamera):
    """
    Camera class for rendering the scene.
    """

    def __init__(self, scene: BaseSimScene) -> None:
        # Initialize camera properties
        self._init_pos = (0.5, 0.5, 0.5)
        self._init_lookat = (0.2, 0.0, 0.2)
        self._init_fov = 45

        #
        self._camera = scene.scene.add_camera(
            pos=self._init_pos,
            lookat=self._init_lookat,
            fov=self._init_fov,
            GUI=True,
        )

    def render(self) -> dict:
        """
        Render the scene from the camera's perspective.
        """
        rgb, depth, _, _ = self._camera.render()
        return {"rgb": rgb, "depth": depth}

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Get the resolution of the camera.
        """
        return self._camera.resolution
