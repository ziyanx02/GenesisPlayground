from robot_display.utils.robot import Robot

class Display(Robot):
    def __init__(self, cfg, vis_options=None):
        self.cfg = cfg

        # if "control" not in self.cfg.keys():
        #     self.cfg["control"] = {"control_freq": 50}
        # if "foot_names" not in self.cfg["robot"].keys():
        #     self.cfg["robot"]["foot_names"] = []
        # if "links_to_keep" not in self.cfg["robot"].keys():
        #     self.cfg["robot"]["links_to_keep"] = []
        
        super().__init__(
            asset_file=self.cfg.robot_args.morph_args.file,
            foot_names=[self.cfg.robot_args.left_foot_link_name, self.cfg.robot_args.right_foot_link_name],
            links_to_keep=self.cfg.robot_args.morph_args.links_to_keep,
            scale=self.cfg.robot_args.morph_args.scale,
            fps=self.cfg.robot_args.ctrl_freq,
            vis_options=vis_options,
        )

        if "body_name" in vars(self.cfg.robot_args).keys():
            self.set_body_link(self.get_link_by_name(self.cfg.robot_args.body_name))
        if "dof_names" in vars(self.cfg.robot_args).keys():
            assert len(self.cfg.robot_args.dof_names) == self.num_dofs, "Number of dof names should match the number of dofs"
            self.set_dof_order(self.cfg.robot_args.dof_names)

    def update(self):
        self.step_vis()