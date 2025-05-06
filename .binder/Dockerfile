FROM dolfinx/dolfinx:stable

RUN python3 -m pip install matplotlib numpy gmsh pyvista
COPY main.py /home/fenics/main.py
CMD ["python3", "/home/fenics/main.py"]
