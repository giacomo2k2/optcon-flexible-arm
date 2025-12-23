# Animation of the optimal trajectory
if visu_animation:
    anim.animate_comparison(model, xx_star, xx_ref, eq_pair1, eq_pair2, target_fps=30)