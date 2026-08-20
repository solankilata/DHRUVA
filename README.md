# DHRUVA — Slip-Risk-Aware Terrain Perception and Confidence-Calibrated Path Planning for Lunar Rover Navigation

*Detection of Hazards & Risk through Unified Vision-based Autonomy*

## Overview

DHRUVA is an AI pipeline that helps a planetary rover decide **where it's safe to drive**, not just what the terrain looks like. Given a rover camera image, it:

1. **Classifies terrain** (soil, bedrock, sand, big rock) using a trained U-Net segmentation model
2. **Estimates slip/entrapment risk** for each region, grounded in documented rover mobility incidents (e.g. NASA's Spirit rover sand entrapment)
3. **Quantifies prediction uncertainty** using Monte Carlo Dropout, flagging regions where the model is unsure
4. **Plans the safest path** between two points using a risk-weighted A* search, avoiding high-risk terrain while flagging low-confidence path segments for human review

Built as a B.Sc. (Hons.) Data Science & AI term project, contextually framed around India's lunar south pole exploration program (Chandrayaan-3/4).

## Results

- **Terrain segmentation:** mIoU 0.81, macro F1 0.88 on held-out validation data (full metrics in `outputs/`)
- **Risk-aware path planning:** compared against a naive shortest-path baseline, DHRUVA's planned routes reduce mean risk exposure by **22.2%** and peak risk by **15.3%**, at a 27% path-length cost — see `outputs/baseline_comparison.png`

## Project Structure