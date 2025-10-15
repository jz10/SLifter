ARCH      = sm_75
#===============================================================================
# Targets to Build
#===============================================================================

build: setup.py
	python setup.py build develop

clean:
	rm -rf *.egg-info *.so build __pycache__

run: build
	$(LAUNCHER) python main.py
	$(LAUNCHER) python main.py

sass: $(program)
	cuobjdump --dump-sass ./$(program) > convolutionDeformable-cuda.sass