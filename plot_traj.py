import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16})
# plt.style.use('dark_background')


def rmse(truth, predicted):
    if truth.shape != predicted.shape:
        raise ValueError("Input arrays must have same shape")

    error = np.linalg.norm(truth - predicted, axis=1)

    squared_diff = (error) ** 2
    mse = np.mean(squared_diff)
    root_mse = np.sqrt(mse)

    return root_mse


def multiply_quaternions(q1, q2):
    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]
    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z]).transpose()


if __name__ == "__main__":
    path = "output/horse/"
    # path = "../gaussian-splatting-w-pose/output/room0_0sh/"
    traj_track = np.load(path + "tracking_traj.npy")
    traj_gt = np.load(path + "tracking_traj_gt.npy")

    fig1 = plt.figure(figsize=(6.4 * 1.5, 4.8 * 1.5))
    ax1 = plt.axes()
    ax1 = fig1.add_subplot(1, 1, 1, projection="3d")

    ax1.set_xlabel("Meters (m)")
    ax1.set_title("Tracking Trajectory")

    fig1.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.xaxis.label.set_color("black")
    ax1.tick_params(axis="x", colors="black")
    ax1.yaxis.label.set_color("black")
    ax1.tick_params(axis="y", colors="black")
    fig1.set_tight_layout(True)
    # ax1.grid(False)

    print(traj_gt.shape)
    print(traj_track.shape)

    N = 1000

    ax1.plot(
        traj_gt[:N, 4],
        traj_gt[:N, 5],
        traj_gt[:N, 6],
        color="red",
        label="Ground truth",
    )
    ax1.plot(
        traj_track[:N, 4],
        traj_track[:N, 5],
        traj_track[:N, 6],
        color="blue",
        label="Tracked",
    )
    ax1.legend()

    fig1.tight_layout()

    fig1.savefig("tracking.png", bbox_inches="tight", pad_inches=0.01, dpi=150)

    # TODO: plot absolute trajectory error
    fig2 = plt.figure(2)
    ax2 = plt.axes()
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Meters (m)")
    ax2.set_title("Absolute Trajectory Error")
    ax2.plot(np.linalg.norm(traj_track[:N, 4:] - traj_gt[:N, 4:], axis=1))

    # TODO: plot orientation error
    fig3 = plt.figure(3)
    ax3 = plt.axes()
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Angle (deg)")
    ax3.set_title("Orientation Error")
    gtconj = np.array(
        [traj_gt[:N, 0], -traj_gt[:N, 1], -traj_gt[:N, 2], -traj_gt[:N, 3]]
    ).transpose()
    diff = multiply_quaternions(traj_track[:N, :4], gtconj)
    diff = diff / np.expand_dims(np.linalg.norm(diff, axis=1), axis=1)
    o_err = 2 * np.arccos(abs(diff[:N, 0]))
    o_err_deg = np.rad2deg(o_err)  # Convert to degrees
    ax3.plot(o_err_deg)
    # Return rotation angle of the difference quaternion
    # diff = diff / torch.norm(diff)
    # return 2 * torch.acos(abs(diff[0]))

    # Print RMSE
    print(rmse(traj_gt[:, 4:], traj_track[:, 4:]))

    plt.show()
