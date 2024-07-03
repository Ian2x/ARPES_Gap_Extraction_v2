# Advanced Fitting Methods for Small Energy Gaps

This repository contains the code and methodology for implementing advanced 1.5D and 2D fitting methods aimed at improving the extraction of small energy gaps, such as those in novel superconductors. This README provides an overview of the project's motivation, methodology, and future directions.

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Proof of Concept](#proof-of-concept)
- [Testing on Real Data](#testing-on-real-data)
- [Comparison of Methods](#comparison-of-methods)
- [Future Efforts](#future-efforts)

## Introduction
### The Goal
The primary goal of this project is to extract small energy gaps, which are crucial for research into novel superconductors. These gaps are important for understanding phenomena such as:
- Phase transitions
- Superconductivity fluctuations
- Giant superconducting fluctuations in compensated semimetals
- Topological superconductors

### Existing Problems
Current methods, such as EDC/MDC fits, have significant limitations:
- Require good knowledge of self-energy and background
- Complicated by detector momentum and energy resolutions, low signal-to-noise ratio, or intensity modulation/broadening
- Fail to fully utilize 2D data space

## Motivation
### Conventional Fit Flaws
Conventional 1D fits have several weaknesses:
- Fail to fully utilize available data, ignoring most EDCs
- Peaks near kf merge and are hard to differentiate
- Peaks away from kf show little movement

### Advanced Fitting Methods
1.5D and 2D fitting methods aim to overcome these weaknesses by using more data from varied momenta, thus improving gap extraction quality.

![temp](https://github.com/Ian2x/ARPES_Gap_Extraction_v2/assets/72709097/9103b065-84bd-4622-b395-f86c37bf8a89)

## Proof of Concept
### Utilizing Multiple EDCs on Simulated Data
We tested three different fit methods on simulated data:
1. **1D Fit**
2. **1.5D Fit**
3. **2D Fit**

#### Methodology
- **1.5D Fit:** Fits multiple EDCs separately, making the fit more resilient to local minima and faster than 2D fits.
- **2D Fit:** Fits all EDCs at once, providing more comprehensive results but requiring more computational resources.

#### Results
- **1.5D Fit:** Demonstrated substantial improvement on simulated results, reducing residuals across the parameter space.
- **2D Fit:** Showed visually better fits with low residuals, but had two local minima for the gap due to being underconstrained.

![temp](https://github.com/Ian2x/ARPES_Gap_Extraction_v2/assets/72709097/abfa1ac6-87f3-431b-8da3-a3e12ba81153)

## Testing on Real Data
We applied the experimental strategy to real data, showcasing results for:
- Extracted gaps compared to symmetrized Norman (value + error bar v. temperature)
- More realistic data simulations to test fits

![temp](https://github.com/Ian2x/ARPES_Gap_Extraction_v2/assets/72709097/8402e3fa-20a6-49ca-be40-6c57e86287e6)

## Comparison of Methods
### 2D Fit
- **Pros:** Visually appealing, low residuals
- **Cons:** Underconstrained, leading to multiple local minima

### 1.5D Fit
- **Pros:** Robust gap size estimates, fast, better at handling low resolution
- **Cons:** May struggle with overlapping bands

### 1D Fit
- **Pros:** Simple, very fast
- **Cons:** High variance, especially for low resolution data

![temp](https://github.com/Ian2x/ARPES_Gap_Extraction_v2/assets/72709097/8c5bfb31-ab60-45a3-a434-6a2f0c6806d0)

## Future Efforts
### Improvements
- Explore different parameter groupings when fitting
- Improve constraints for 2D fits
- Account for overlapping bands

### Applications
- Better small gap extraction
- Improved noise tolerance for more rapid data collection
- Labeling data for CNNs due to fast processing

## Conclusion
The advanced 1.5D and 2D fitting methods present significant improvements over conventional 1D fits, particularly in handling low resolution and noisy data. Future efforts will focus on refining these methods and exploring their broader applications in extracting small energy gaps in superconductors and beyond.

For more detailed information on the methodology and results, please refer to the corresponding sections in the repository.

---

**Primary References:**
- [Phenomenology of the low-energy spectral function in high-Tc superconductors](http://ex7.iphy.ac.cn/downfile/32_PRB_57_R11093.pdf)
- [BCS-like Bogoliubov Quasiparticles in High-Tc Superconductors Observed by Angle-Resolved Photoemission Spectroscopy](https://arxiv.org/pdf/cond-mat/0304505.pdf)
- [Topological superconductors: a review](https://arxiv.org/pdf/1608.03395.pdf)
