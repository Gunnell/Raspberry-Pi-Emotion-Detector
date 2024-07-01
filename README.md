# Raspberry Pi Emotion Detector

This project is an Emotion Detection System that uses a Raspberry Pi to capture and analyze facial expressions, displaying the detected emotions on a 7-segment display.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Setup Instructions](#setup-instructions)
- [License](#license)

## Introduction

The Raspberry Pi Emotion Detector captures images from a webcam, detects faces, and uses a trained model to identify emotions. The detected emotions are displayed on a 7-segment display using GPIO pins.

## Features

- Real-time emotion detection
- Displays emotions on a 7-segment display
- Supports emotions: Anger, Disgust, Fear, Happiness, Neutrality, Sadness, Surprise

## Hardware Requirements

- Raspberry Pi 4 (recommended) or Raspberry Pi 2
- USB Webcam
- 7-Segment Display
- Connecting wires
- Breadboard
- Resistors (if needed)
- Internet connection for initial setup

## Software Requirements

- Raspbian OS (latest version)
- Python 3


## Setup Instructions

### 1. Install Raspbian OS

Follow the official Raspberry Pi documentation to install Raspbian OS on your Raspberry Pi.

### 2. Update and Upgrade

Open a terminal and run:

```sh
sudo apt-get update
sudo apt-get upgrade
