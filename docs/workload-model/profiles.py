# -*- coding: utf-8 -*-
"""The eighteen mission profiles, as explicit sensor configurations and update rates."""
from pipeline import Mission

M = Mission
PROFILES = [

# ------------------------------------------------------------------ EDGE AI
M('Edge AI device', 'Event detection & classification', 2,
  'One 1080p camera at 15 Hz, tiny detector, no map, no closed loop', 200,
  mono=(1, 1920, 1080, 15), det_hz=15, det_gmac=1.6, det_mpar=3.2),

M('Edge AI device', 'Multi-stream tracking & anomaly', 5,
  'Four 1080p streams at 15 Hz, small detector plus re-identification', 200,
  mono=(4, 1920, 1080, 15), det_hz=60, det_gmac=5.0, det_mpar=9),

M('Edge AI device', 'Supervisory control & command grounding', 8,
  'Two 1080p streams, 0.5 B on-device VLM at 1 query/s, operator in the loop', 500,
  mono=(2, 1920, 1080, 15), det_hz=30, det_gmac=5.0, det_mpar=9,
  vlm_qps=1.0, vlm_par=0.5, vlm_prefill=400, vlm_decode=48, vlm_vis_gop=40,
  replan_hz=1),

# ------------------------------------------------------------------ DRONE
M('Drone', 'Inspection (structure / asset)', 25,
  'Low speed, close standoff, onboard defect detection, GNSS-denied', 120,
  stereo=(1, 1440, 1080, 15, 96), mono=(1, 1440, 1080, 15),
  lidar_pts_s=0.4e6, vio=(2, 1440, 1080, 20), ba_hz=10,
  tsdf_pts_s=0.4e6, esdf_hz=4, det_hz=15, det_gmac=45,
  gain_evals_s=300, replan_hz=2, sdfenc_hz=10, policy_hz=10,
  mpc_hz=50, mpc_dims=(12, 4, 30), cbf_hz=50, ctrl_hz=400, dof=4),

M('Drone', 'ISR (endurance / wide-area search)', 25,
  'Companion-annex far-flight regime: onboard 2 B VLM, emissions-restricted', 200,
  stereo=(1, 1440, 1080, 20, 128), mono=(2, 1440, 1080, 20),
  radar=(1, 20, 12, 512, 128), lidar_pts_s=0.6e6, vio=(2, 1440, 1080, 20), ba_hz=20,
  tsdf_pts_s=0.6e6, esdf_hz=5, det_hz=45, det_gmac=45,
  vlm_qps=2.0, gain_evals_s=600, replan_hz=2, sdfenc_hz=20, policy_hz=20,
  mpc_hz=50, mpc_dims=(12, 4, 30), cbf_hz=50, ctrl_hz=100, dof=4),

M('Drone', 'Interceptor (terminal engagement)', 25,
  'Companion-annex air-superiority regime: 150-250 m/s closure, no VLM in the loop', 80,
  stereo=(1, 1440, 1080, 40, 128), mono=(1, 1440, 1080, 40),
  radar=(1, 20, 32, 1024, 256), lidar_pts_s=1.0e6, vio=(2, 1440, 1080, 40), ba_hz=50,
  tsdf_pts_s=1.0e6, esdf_hz=10, det_hz=100, det_gmac=45,
  replan_hz=1, sdfenc_hz=100, policy_hz=100,
  mpc_hz=100, mpc_dims=(12, 4, 30), cbf_hz=100, ctrl_hz=400, dof=4),

# ------------------------------------------------------------------ QUADRUPED
M('Quadruped', 'Inspection (plant walkdown)', 30,
  'Known map, gauge and thermal reading, 12-DoF whole-body control', 100,
  stereo=(1, 1280, 720, 30, 96), mono=(2, 1280, 720, 30),
  lidar_pts_s=0.3e6, vio=(2, 1280, 720, 30), ba_hz=10,
  tsdf_pts_s=0.3e6, esdf_hz=5, det_hz=30, det_gmac=45,
  gain_evals_s=200, replan_hz=2, sdfenc_hz=20, policy_hz=50,
  mpc_hz=200, mpc_dims=(24, 12, 20), cbf_hz=200, cbf_haz=16, ctrl_hz=1000, dof=12),

M('Quadruped', 'Surveillance (persistent patrol)', 40,
  '360-degree camera ring, person detection and re-identification, day/night', 100,
  stereo=(1, 1280, 720, 30, 96), mono=(4, 1280, 720, 30),
  lidar_pts_s=0.4e6, vio=(2, 1280, 720, 30), ba_hz=15,
  tsdf_pts_s=0.4e6, esdf_hz=5, det_hz=60, det_gmac=45,
  gain_evals_s=400, replan_hz=2, sdfenc_hz=25, policy_hz=50,
  mpc_hz=200, mpc_dims=(24, 12, 20), cbf_hz=200, cbf_haz=16, ctrl_hz=1000, dof=12),

M('Quadruped', 'ISR (dismounted, comms-denied)', 60,
  'Unstructured terrain, onboard 2 B VLM, no datalink assumption', 150,
  stereo=(1, 1440, 1080, 30, 128), mono=(4, 1280, 720, 30),
  radar=(1, 10, 12, 512, 128), lidar_pts_s=0.6e6, vio=(2, 1440, 1080, 30), ba_hz=20,
  tsdf_pts_s=0.6e6, esdf_hz=8, det_hz=70, det_gmac=45,
  vlm_qps=1.0, gain_evals_s=1500, replan_hz=3, sdfenc_hz=30, policy_hz=50,
  mpc_hz=250, mpc_dims=(24, 12, 20), cbf_hz=250, cbf_haz=16, ctrl_hz=1000, dof=12),

# ------------------------------------------------------------------ AMR
M('AMR', 'Warehousing (structured aisles)', 30,
  'Known map, fiducials, 2 m/s, near-static environment', 150,
  stereo=(1, 1280, 720, 20, 64), mono=(1, 1280, 720, 20),
  lidar_pts_s=0.15e6, vio=(2, 1280, 720, 20), ba_hz=8,
  tsdf_pts_s=0.15e6, esdf_hz=4, det_hz=20, det_gmac=22,
  replan_hz=2, sdfenc_hz=10, policy_hz=20,
  mpc_hz=50, mpc_dims=(8, 3, 25), cbf_hz=100, cbf_haz=8, ctrl_hz=200, dof=3),

M('AMR', 'Logistics (mixed / dynamic yard)', 40,
  'Humans and vehicles present, indoor-outdoor transitions, 4 m/s', 120,
  stereo=(1, 1280, 720, 30, 96), mono=(3, 1280, 720, 30),
  radar=(1, 10, 12, 512, 128), lidar_pts_s=0.5e6, vio=(2, 1280, 720, 30), ba_hz=15,
  tsdf_pts_s=0.5e6, esdf_hz=8, det_hz=45, det_gmac=45,
  gain_evals_s=300, replan_hz=3, sdfenc_hz=25, policy_hz=40,
  mpc_hz=100, mpc_dims=(8, 3, 25), cbf_hz=200, cbf_haz=12, ctrl_hz=200, dof=3),

M('AMR', 'Loading / unloading (manipulation)', 75,
  'Instance segmentation and 6-DoF pose, chunked grasp policy, 6-DoF arm', 100,
  stereo=(2, 1280, 720, 30, 96), mono=(3, 1280, 720, 30),
  lidar_pts_s=0.4e6, vio=(2, 1280, 720, 30), ba_hz=15,
  tsdf_pts_s=0.6e6, esdf_hz=10, det_hz=45, det_gmac=70, det_mpar=42,
  vla_hz=10, vla_par=3, replan_hz=5, sdfenc_hz=30, policy_hz=40,
  mpc_hz=200, mpc_dims=(18, 9, 20), cbf_hz=250, cbf_haz=16, ctrl_hz=1000, dof=9),

# ------------------------------------------------------------------ HUMANOID
M('Humanoid', 'Industrial automation (structured cell)', 60,
  'Repetitive task set, 30-DoF whole-body MPC at 500 Hz', 80,
  stereo=(1, 1280, 720, 30, 96), mono=(3, 1280, 720, 30),
  lidar_pts_s=0.2e6, vio=(2, 1280, 720, 30), ba_hz=20,
  tsdf_pts_s=0.3e6, esdf_hz=10, det_hz=40, det_gmac=70, det_mpar=42,
  vla_hz=10, vla_par=3, replan_hz=5, sdfenc_hz=30, policy_hz=50,
  mpc_hz=500, mpc_dims=(60, 30, 20), cbf_hz=500, cbf_haz=20, ctrl_hz=1000, dof=30),

M('Humanoid', 'Cobot (human-adjacent, contact-rich)', 90,
  'Human pose and intent prediction, contact scheduling, 1 kHz safety filter', 60,
  stereo=(2, 1280, 720, 30, 96), mono=(4, 1280, 720, 30),
  lidar_pts_s=0.3e6, vio=(2, 1280, 720, 30), ba_hz=25,
  tsdf_pts_s=0.4e6, esdf_hz=15, det_hz=60, det_gmac=70, det_mpar=42,
  vla_hz=20, vla_par=3, replan_hz=5, sdfenc_hz=50, policy_hz=100,
  mpc_hz=500, mpc_dims=(60, 30, 20), cbf_hz=1000, cbf_haz=24, ctrl_hz=2000, dof=30),

M('Humanoid', 'House work (open-world, long-horizon)', 150,
  'Open-vocabulary tasking, 3 B VLA at 30 Hz, 2 B VLM task planner', 100,
  stereo=(2, 1440, 1080, 30, 128), mono=(4, 1280, 720, 30),
  lidar_pts_s=0.4e6, vio=(2, 1440, 1080, 30), ba_hz=30,
  tsdf_pts_s=0.6e6, esdf_hz=20, det_hz=60, det_gmac=70, det_mpar=42,
  vlm_qps=1.0, vla_hz=30, vla_par=3, gain_evals_s=800, replan_hz=8,
  sdfenc_hz=50, policy_hz=100,
  mpc_hz=500, mpc_dims=(60, 30, 20), cbf_hz=1000, cbf_haz=24, ctrl_hz=2000, dof=30),

# ------------------------------------------------------------------ AV
M('Autonomous vehicle', 'SAE L2 / L2+ (partial automation)', 30,
  'Hands-on. Six 2 MP cameras at 30 Hz, five radars, no LiDAR, driver is fallback', 150,
  mono=(6, 1928, 1208, 30), radar=(5, 20, 12, 512, 128),
  vio=(2, 1928, 1208, 30), ba_hz=10,
  det_hz=30, det_gmac=15, det_mpar=30, replan_hz=10,
  mpc_hz=50, mpc_dims=(8, 2, 30), cbf_hz=100, cbf_haz=10, ctrl_hz=100, dof=2),

M('Autonomous vehicle', 'SAE L3 (conditional automation)', 100,
  'Eyes-off in ODD. Eleven cameras, one LiDAR, five radars, multi-agent prediction', 100,
  stereo=(1, 1928, 1208, 30, 128), mono=(11, 1928, 1208, 30),
  radar=(5, 20, 12, 512, 128), lidar_pts_s=1.2e6, vio=(2, 1928, 1208, 30), ba_hz=25,
  tsdf_pts_s=1.2e6, esdf_hz=20, det_hz=30, det_gmac=75, det_mpar=45,
  replan_hz=20, sdfenc_hz=50, policy_hz=50,
  mpc_hz=100, mpc_dims=(8, 2, 30), cbf_hz=200, cbf_haz=16, ctrl_hz=200, dof=2),

M('Autonomous vehicle', 'SAE L4 / L5 (high / full automation)', 800,
  'No fallback driver. 20+ cameras to 8 MP, 3-5 LiDAR, 8 radars, dual-redundant stack', 100,
  stereo=(2, 1928, 1208, 30, 128), mono=(20, 2896, 1876, 30),
  radar=(8, 20, 32, 1024, 256), lidar_pts_s=4.0e6, vio=(4, 1928, 1208, 30), ba_hz=50,
  tsdf_pts_s=4.0e6, esdf_hz=40, det_hz=60, det_gmac=155, det_mpar=95,
  vlm_qps=0.25, vlm_par=7, gain_evals_s=500, replan_hz=40,
  sdfenc_hz=100, policy_hz=100,
  mpc_hz=200, mpc_dims=(8, 2, 30), cbf_hz=400, cbf_haz=24, ctrl_hz=400, dof=2),
]

ROWS = [p.summary() for p in PROFILES]
