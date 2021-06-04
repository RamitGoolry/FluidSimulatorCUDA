default: compile run

compile:
	nvcc main.cu -lglut -lGLU -lGL -o NavierStokes

run:
	./NavierStokes
