# Visual Inertial Odometry Algorithms and Platforms

## Assessment of OpenVINS for an embedded multi-cam drone in 2026

OpenVINS is no longer the right base to start from in 2026. It is still an excellent reference implementation of MSCKF — clean, well-documented, type-safe covariance management — but it has been effectively in maintenance mode since 2023 and the same lab has already published a successor. For an embedded multi-camera drone build in 2026, you should not branch from `rpng/open_vins`. The reasonable shortlist is `rpng/MINS`, `ethz-mrl/okvis2`, and (if you have NVIDIA silicon in the loop) `nvidia-isaac/cuVSLAM`. Below is why, with the trade-offs spelled out.

### State of OpenVINS itself

The repository confirms what you suspected. The last tagged release is **v2.7 (June 20, 2023)**. The README "News" feed stops in May 2023. There are roughly 58 open issues and 4 open PRs sitting unaddressed. The codebase is GPL-3.0, which is fine for research but a hard problem if you ever want to ship a closed product on the drone. The "arbitrary number of cameras" feature you'll see advertised in the feature list applies primarily to the simulator and the EKF state machinery; the tracker is documented as mono / stereo / synchronized-binocular, and getting genuine ≥3 camera tracking to behave well requires non-trivial work on the front-end (the VOXL2 community has done it, but it is not a polished, supported path). The algorithm itself — sliding-window MSCKF with a few SLAM features — is now well behind state-of-the-art optimization-based systems on accuracy, and behind newer filter-based systems (SchurVINS, MINS) on the accuracy/compute trade-off it was originally optimized for.

The strongest argument for OpenVINS in 2026 is pedagogical: the type-based state, Jacobians, and observability derivations are still some of the cleanest in the open-source world. If your goal is to *understand* MSCKF and then write your own, reading OpenVINS is still worthwhile. If your goal is a production drone stack, that is a different question.

### The successor from the same group: MINS

This is the part most people miss. The OpenVINS authors (Geneva, Lee, Chen, Huang at UDel RPNG) published **MINS** in 2023 and the journal version landed in *Journal of Field Robotics* in 2025. MINS is explicitly described as building on OpenVINS' core and superseding their earlier MIMC-VINS for multi-sensor work. Concretely it adds proper multi-camera + multi-IMU support, optional LiDAR / GNSS / wheel fusion, a high-order on-manifold state interpolation, and a dynamic cloning strategy aimed at keeping compute bounded as you add cameras. For an embedded multi-camera drone, this is the natural OpenVINS upgrade.

Caveats, since you asked me not to flatter: MINS is **ROS1 only** (Melodic / Noetic Dockerfiles), still GPL-3.0, has only ~38 commits and 16 open issues, and the third-party LiDAR dependency (libpointmatcher) drags in baggage you don't need for VIO. The ROS1 problem is real — ROS1 Noetic is at end-of-life and any new 2026 drone stack should be on ROS2. You will be porting. But MINS gets you the algorithm work the OpenVINS team did *after* 2023, which is exactly the gap you noticed.

### OKVIS2 (ETH MRL / TUM, Leutenegger)

If you can tolerate (or prefer) an **optimization-based** back-end, OKVIS2 is the most credible choice today. License is **BSD-3**, which removes the commercial-use friction. It explicitly supports N cameras (tested with mono, stereo, and four-camera rigs) plus IMU, has built-in loop closure, ships a ROS2 build target, and a working Realsense D455 demo. The same group released **OKVIS2-X** in October 2025, adding optional dense depth, LiDAR, and GNSS — and in their own evaluation it tops EuRoC and Hilti22 against current alternatives. Independent reading of recent benchmark papers puts OKVIS2 and Basalt at the top of optimization-based VIO accuracy on EuRoC, generally ahead of OpenVINS.

The catch is compute. Sliding-window optimization with Ceres / SuiteSparse on an aggressive-motion drone with three cameras is heavier than an MSCKF filter and was traditionally what pushed people toward OpenVINS for tight SWaP. On modern ARM (Cortex-A78 class, Jetson Orin Nano / NX, Snapdragon Flight RB5 / VOXL2) this is no longer the killer it was in 2019; OKVIS2 runs real-time on those targets in stereo-inertial mode. With ≥3 cameras you will want to tune keyframe rate and window length, and CNN-based sky masking (optional, requires LibTorch) is an extra ~1 W you can disable.

The other caveat is that OKVIS2 publication / write access on GitHub is admin-gated and "research software" — meaning if you find a bug you'll fix it yourself rather than getting a fast upstream merge.

### Basalt (Usenko, TUM → Luxonis)

Also **BSD-3**. Multi-camera + fisheye + IMU; the front-end uses KLT tracking on a lightweight feature set, the back-end is a square-root Schur-based BA. Of the optimization-based options it has historically been the *fastest* — that is the reputation it has, and the OpenVINS paper itself reports Basalt as the only system that beat OpenVINS' stereo numbers when v2 was released. Active development happens on GitLab; Luxonis maintains a fork and uses it inside their OAK ecosystem. This is the choice if you want lean compute *and* optimization-style accuracy, and care less about an integrated loop-closure / map-merging stack than OKVIS2 gives you. The downsides: smaller community than OKVIS2/ORB-SLAM3, documentation is thinner, and there is no official ROS2 maintenance — you'll be using a fork.

### NVIDIA Isaac cuVSLAM / Elbrus

If your drone has — or might have — a **Jetson Orin Nano / NX / AGX** on board, this is a serious option you should not skip. NVIDIA published a 2025 paper (arXiv 2506.04359) reporting <5 cm mean position error on EuRoC stereo-inertial and demonstrating real-time operation with up to 4 stereo pairs (8 cameras) on Jetson AGX Orin. It supports IMU fusion, multi-cam (up to 32), is shipped under Apache 2.0 in the Isaac ROS stack (commercial-friendly), and has actual industrial backing. The price you pay is hard NVIDIA lock-in: no CUDA, no cuVSLAM. If you choose a non-NVIDIA SoC for the drone (e.g. Qualcomm RB5, Ambarella, NXP), this is off the table. If your roadmap is Jetson, branching off this is by far the lowest-effort path to a flying robot with multi-camera VIO that works.

### Others worth knowing but probably not the base

ORB-SLAM3 (GPLv3 with paid commercial option) is the most accurate generally-available open VIO on EuRoC stereo-inertial (~3.5 cm) and supports multi-map + fisheye + IMU, but ORB descriptor extraction is heavy on ARM and the codebase has not seen serious upstream changes in years; people use it as a benchmark, not as a base anymore. SchurVINS (CVPR 2024) is interesting *as ideas*: it shows you can shave ~50% off the compute of comparable optimization-based VIO with a Schur-complement filter while matching accuracy, and the authors specifically target resource-constrained devices. The code is academic and forked into `Jas0nG/ov_SchurVINS`; it's not a production base, but it is a paper to read before you commit. Kimera-VIO (BSD, MIT SPARK) has a 2024 multi-camera extension and an active ROS2 wrapper, but the project's center of gravity is metric-semantic mapping (Kimera2), not minimal-footprint drone VIO. DM-VIO is monocular by design — irrelevant to your multi-cam plan.

### Comparison matrix

| System | License | Back-end | Multi-cam (≥3) | Loop closure | ROS2 | Active in 2024-26 | Embedded fit |
|---|---|---|---|---|---|---|---|
| OpenVINS | GPL-3 | MSCKF filter | Partial (sim + state; tracker is mono/stereo) | No (loose, via ov_secondary) | Yes, but stale | No (last release Jun 2023) | Was strong; now mid-tier |
| MINS (same authors) | GPL-3 | Filter + on-manifold interp. | **Yes**, multi-cam + multi-IMU | No native; via mapping | **No (ROS1 only)** | Modest; journal pub 2025 | Strong on paper; ROS1 is a problem |
| OKVIS2 | BSD-3 | Sliding-window BA + LC | **Yes** (tested 4-cam) | **Yes** | Yes | **Yes** (OKVIS2-X Oct 2025) | Good on A78/Orin; tune carefully |
| Basalt | BSD-3 | Square-root BA | **Yes** | Partial | Forks only | Maintained on GitLab | **Best lean BA option** |
| cuVSLAM | Apache-2 (Isaac) | CUDA stereo VO + IMU | **Yes** (up to 32) | Yes | Yes | **Yes** (2025 paper, NVIDIA) | **Only on NVIDIA Jetson** |
| ORB-SLAM3 | GPL-3 (commercial avail.) | Full SLAM + IMU | Yes (mono/stereo + IMU + multi-map) | Yes | Community | Stagnant upstream | ORB on ARM is heavy |
| Kimera-VIO | BSD-2 | Stereo VIO + mesh | Recent extension | Yes | Yes | Yes | Mapping focus, not minimal |

### Recommendation

For an embedded, SWaP-constrained, multi-camera drone in 2026, the decision tree is short.

If your compute target is **Jetson Orin** (any variant): branch off **cuVSLAM / Isaac ROS Visual SLAM**. It has the cleanest license, real-time multi-camera up to 4 stereo pairs on Orin AGX is reported, it's actively developed by a vendor, and you avoid reimplementing pieces that NVIDIA has already accelerated.

If your compute target is **non-NVIDIA ARM** (Qualcomm, Ambarella, NXP, Apple, custom) **and** you need a permissive license: branch off **OKVIS2**. It is the closest thing to a 2026-current, BSD-licensed, multi-camera, loop-closing VIO. Plan time to tune the window and front-end for your camera rig and to merge OKVIS2-X improvements if you need depth or GNSS later.

If you are **research-leaning, license-tolerant, and want the lineage of the OpenVINS work without the staleness**: branch off **MINS** from the same authors. You will pay the ROS1→ROS2 porting tax, and you'll be on GPL-3, but you'll get the multi-camera/multi-IMU work the OpenVINS team did after they stopped pushing to `open_vins`.

OpenVINS itself in 2026 is a reference text, not a foundation.

### Things to verify before you commit

A few claims above lean on second-hand benchmark summaries (independent EuRoC RMSE numbers across the systems can vary by ±20% depending on initialization, calibration quality, and evaluation tooling). Before you sink engineering time, I would (a) actually run OKVIS2, MINS, and — if relevant — cuVSLAM end-to-end on a UZH-FPV sequence and your own rig, not just EuRoC, because aggressive drone motion is where filters and tightly-coupled BA diverge in interesting ways; (b) confirm the OKVIS2 multi-camera mode with your specific lens model and overlap geometry, since the "tested with 4 cameras" claim doesn't guarantee any 4-camera geometry; (c) re-check MINS for ROS2 forks before assuming you have to port it yourself; and (d) confirm your IMU intrinsics / time-sync story is in place, because for all of these systems calibration quality is a bigger swing on accuracy than the choice of algorithm.Sources:
- [rpng/open_vins on GitHub](https://github.com/rpng/open_vins)
- [rpng/MINS on GitHub](https://github.com/rpng/MINS)
- [MINS: Efficient and Robust Multisensor-aided INS (J. Field Robotics 2025)](https://onlinelibrary.wiley.com/doi/10.1002/rob.22546)
- [ethz-mrl/okvis2 on GitHub](https://github.com/ethz-mrl/okvis2)
- [OKVIS2-X paper (arXiv 2510.04612, Oct 2025)](https://arxiv.org/abs/2510.04612)
- [Basalt mirror on GitHub](https://github.com/VladyslavUsenko/basalt)
- [Basalt on GitLab (primary)](https://gitlab.com/VladyslavUsenko/basalt)
- [UZ-SLAMLab/ORB_SLAM3 on GitHub](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [lukasvst/dm-vio on GitHub](https://github.com/lukasvst/dm-vio)
- [MIT-SPARK/Kimera-VIO on GitHub](https://github.com/MIT-SPARK/Kimera-VIO)
- [SchurVINS (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_SchurVINS_Schur_Complement-Based_Lightweight_Visual_Inertial_Navigation_System_CVPR_2024_paper.pdf)
- [NVIDIA-ISAAC-ROS/isaac_ros_visual_slam](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)
- [cuVSLAM paper (arXiv 2506.04359, Jun 2025)](https://arxiv.org/abs/2506.04359)
- [Three-Camera Open-VINS Server on the VOXL 2 (Ascend Engineering)](https://www.ascendengineer.com/post/three-camera-open-vins-server-on-the-voxl-2)