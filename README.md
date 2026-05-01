# Software 3D Renderer in Python

A real-time 3D rendering engine written entirely in Python, using Pygame for display and Numba for high‑performance triangle rasterisation – no GPU libraries or any other 3d rendering libraries used.

## Features

- Vertices transformed with custom 4×4 matrix math (translation, rotation, scale, perspective projection).
- Camera movement with mouse look and WASD/QE controls.
- Flat shading with directional light and ambient term.
- Numba‑JIT‑compiled triangle rasteriser for smooth 60 FPS on the CPU.
- Multiple procedurally generated shapes: cube, sphere, cylinder, cone, torus.
- Grid plane and coloured axes for orientation.
- HUD showing FPS and camera position.

## Why I built it

I wanted to understand exactly what happens inside a graphics pipeline, from model transformations to clipping, lighting. Using Numba was a challenge to keep everything in pure Python, yet still fast enough for real‑time interaction.

## how to run

python final_code.py

## controls

    WASD – move

    Q / E – ascend / descend

    Mouse – look around

    Esc – quit
