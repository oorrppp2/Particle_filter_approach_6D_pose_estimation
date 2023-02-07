#include "buffer.hpp"
#include "render.hpp"
#include <iostream>

VertexBuffer::VertexBuffer()
: d_positions(0), d_colors(0), d_normals(0), d_texcoords(0), d_indices(0),
  num_positions(0), num_colors(0), num_normals(0), num_texcoords(0), num_indices(0)
{	
	translation = glm::vec3(0, 0, 0);
	rotation = glm::mat3(1.0f);
}

VertexBuffer::~VertexBuffer() {
	Reset();
}

void VertexBuffer::Reset() {
	FreeCudaArray(d_positions, num_positions);
	FreeCudaArray(d_colors, num_colors);
	FreeCudaArray(d_normals, num_normals);
	FreeCudaArray(d_texcoords, num_texcoords);
}

void VertexBuffer::SetPositions(const std::vector<glm::vec3>& positions) {

	if (num_positions != positions.size()) {
		FreeCudaArray(d_positions, num_positions);
		num_positions = positions.size();
		cudaMalloc((void**)&d_positions, sizeof(glm::vec3) * positions.size());
	}

	cudaMemcpy(d_positions, positions.data(), sizeof(glm::vec3) * positions.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetColors(const std::vector<glm::vec3>& colors) {
	if (num_colors != colors.size()) {
		FreeCudaArray(d_colors, num_colors);
		num_colors = colors.size();
		cudaMalloc((void**)&d_colors, sizeof(glm::vec3) * colors.size());
	}
	cudaMemcpy(d_colors, colors.data(), sizeof(glm::vec3) * colors.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetNormals(const std::vector<glm::vec3>& normals) {
	if (num_normals != normals.size()) {
		FreeCudaArray(d_normals, num_normals);
		num_normals = normals.size();
		cudaMalloc((void**)&d_normals, sizeof(glm::vec3) * normals.size());
	}
	cudaMemcpy(d_normals, normals.data(), sizeof(glm::vec3) * normals.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetTexcoords(const std::vector<glm::vec2>& texcoords) {
	if (num_texcoords != texcoords.size()) {
		FreeCudaArray(d_texcoords, num_texcoords);
		num_texcoords = texcoords.size();
		cudaMalloc((void**)&d_texcoords, sizeof(glm::vec2) * texcoords.size());
	}
	cudaMemcpy(d_texcoords, texcoords.data(), sizeof(glm::vec2) * texcoords.size(), cudaMemcpyHostToDevice);
}

void VertexBuffer::SetIndices(const std::vector<glm::ivec3>& indices) {
	if (num_indices != indices.size()) {
		FreeCudaArray(d_indices, num_indices);
		num_indices = indices.size();
		cudaMalloc((void**)&d_indices, sizeof(glm::ivec3) * indices.size());
	}
	cudaMemcpy(d_indices, indices.data(), sizeof(glm::ivec3) * indices.size(), cudaMemcpyHostToDevice);
}


FrameBuffer::FrameBuffer()
: d_z(0), d_colors(0), d_depth(0), d_vindices(0), d_vweights(0), row(0), col(0)
{}

FrameBuffer::FrameBuffer(int rows, int cols, float _cx, float _cy, float _fx, float _fy)
: d_z(0), d_colors(0), d_depth(0), d_vindices(0), d_vweights(0), row(0), col(0), cx(_cx), cy(_cy), fx(_fx), fy(_fy)
{
	Initialize(rows, cols);
	ClearBuffer();
}

FrameBuffer::~FrameBuffer() {
	free(matching_scores);
	Reset();
}

void FrameBuffer::Create(int _rows, int _cols, float _cx, float _cy, float _fx, float _fy, int _num_particles) {
	cx = _cx;
	cy = _cy;
	fx = _fx;
	fy = _fy;
	num_particles = _num_particles;
	Initialize(_rows, _cols);
	ClearBuffer();
}

void FrameBuffer::Reset() {
	FreeCudaImage(d_depth, row, col);
	FreeCudaImage(d_vindices, row, col);
	FreeCudaImage(d_vweights, row, col);
	FreeCudaImage(d_z, row, col);
	FreeCudaImage(d_colors, row, col);
	FreeCudaImage(d_findices, row, col);
	FreeCudaImage(depth_src_ptr, row, col);
	FreeCudaImage(other_objects_regions_ptr, row, col);
	cudaFree(p_rotation);
	cudaFree(p_translation);
	cudaFree(score_count_ptr);
	cudaFree(inter_count_ptr);
	cudaFree(union_count_ptr);
}

void FrameBuffer::Initialize(int rows, int cols) {
	if (row != rows || col != cols) {
		Reset();
		row = rows;
		col = cols;
		cudaMalloc(&d_z, sizeof(int) * row * col * num_particles);
		cudaMalloc(&d_colors, sizeof(int) * row * col);
		cudaMalloc(&d_findices, sizeof(int) * row * col);
		cudaMalloc(&d_depth, sizeof(float) * row * col * num_particles);
		cudaMalloc(&d_vindices, sizeof(glm::ivec3) * row * col);
		cudaMalloc(&d_vweights, sizeof(glm::vec3) * row * col);
		cudaMalloc((void**)&depth_src_ptr, sizeof(float) * row * col);
		cudaMalloc((void**)&other_objects_regions_ptr, sizeof(bool) * row * col);
		cudaMalloc((void**)&p_rotation, sizeof(glm::mat3) * num_particles);
		cudaMalloc((void**)&p_translation, sizeof(glm::vec3) * num_particles);
		cudaMalloc(&score_count_ptr, sizeof(unsigned int) * 256 * num_particles);
		cudaMalloc(&inter_count_ptr, sizeof(unsigned int) * 256 * num_particles);
		cudaMalloc(&union_count_ptr, sizeof(unsigned int) * 256 * num_particles);
	
	}
}

void FrameBuffer::ClearBuffer() {
	cudaMemset(d_z, 0, sizeof(int) * row * col * num_particles);
	cudaMemset(d_depth, 0, sizeof(float) * row * col * num_particles);
	cudaMemset(d_findices, 0, sizeof(int) * row * col);
	cudaMemset(d_vweights, 0, sizeof(glm::vec3) * row * col);
	cudaMemset(d_vindices, 0, sizeof(glm::ivec3) * row * col);
	cudaMemset(d_colors, 0, sizeof(int) * row * col);
	cudaMemset(score_count_ptr, 0, sizeof(unsigned int) * 256 * num_particles);
	cudaMemset(inter_count_ptr, 0, sizeof(unsigned int) * 256 * num_particles);
	cudaMemset(union_count_ptr, 0, sizeof(unsigned int) * 256 * num_particles);
}
/*
cv::Mat FrameBuffer::GetImage() {
	cv::Mat res(row, col, CV_8UC4);
	cudaMemcpy(res.data, d_colors, sizeof(int) * row * col, cudaMemcpyDeviceToHost);
	return res;
}
*/
void FrameBuffer::GetDepth(float* depth) {
	cudaMemcpy(depth, d_depth, sizeof(float) * row * col * num_particles, cudaMemcpyDeviceToHost);
}

void FrameBuffer::SetSrcDepth(float* depth_src, bool* other_objects_regions) {
	cudaMemcpy(depth_src_ptr, depth_src, sizeof(float) * row * col, cudaMemcpyHostToDevice);
	cudaMemcpy(other_objects_regions_ptr, other_objects_regions, sizeof(bool) * row * col, cudaMemcpyHostToDevice);
}

void FrameBuffer::GetVMap(glm::ivec3* vindices, glm::vec3* vweights, int* findices) {
	cudaMemcpy(vindices, d_vindices, sizeof(glm::ivec3) * row * col, cudaMemcpyDeviceToHost);
	cudaMemcpy(vweights, d_vweights, sizeof(glm::vec3) * row * col, cudaMemcpyDeviceToHost);
	cudaMemcpy(findices, d_findices, sizeof(int) * row * col, cudaMemcpyDeviceToHost);
}

void FrameBuffer::SetParticleNum(int _num_particles) {
	free(matching_scores);
	matching_scores = new float[_num_particles];
	memset(matching_scores, 0, _num_particles * sizeof(float));
	num_particles = _num_particles;
}

void FrameBuffer::SetThreshold(float threshold) {
	c_threshold = threshold;
}

void FrameBuffer::GetMatchingScore(float* score_buffer) {
	memcpy(score_buffer, matching_scores, sizeof(float) * num_particles);

}