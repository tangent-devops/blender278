
set(INC
	..
	../../glew-mx
)

set(INC_SYS
	${GLEW_INCLUDE_DIR}
)

set(SRC
	attribute.cpp
	background.cpp
	bake.cpp
	buffers.cpp
	camera.cpp
	constant_fold.cpp
	coverage.cpp
	film.cpp
	graph.cpp
	image.cpp
	integrator.cpp
	light.cpp
	mesh.cpp
	mesh_displace.cpp
	mesh_subdivision.cpp
	nodes.cpp
	object.cpp
	osl.cpp
	particles.cpp
	curves.cpp
	scene.cpp
	session.cpp
	shader.cpp
	sobol.cpp
	svm.cpp
	tables.cpp
	tile.cpp
	volume.cpp
)

set(SRC_HEADERS
	attribute.h
	bake.h
	background.h
	buffers.h
	camera.h
	constant_fold.h
	coverage.h
	film.h
	graph.h
	image.h
	integrator.h
	light.h
	mesh.h
	nodes.h
	object.h
	osl.h
	particles.h
	curves.h
	scene.h
	session.h
	shader.h
	sobol.h
	svm.h
	tables.h
	tile.h
	volume.h
)

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

add_definitions(${GL_DEFINITIONS})

add_library(cycles_render ${SRC} ${SRC_HEADERS})
