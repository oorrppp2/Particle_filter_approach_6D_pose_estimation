#ifndef BUFFER_H_
#define BUFFER_H_

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>
//#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

class VertexBuffer
{
public:
	VertexBuffer();
	~VertexBuffer();

	void Reset();

	void SetPositions(const std::vector<glm::vec3>& positions);
	void SetNormals(const std::vector<glm::vec3>& normals);
	void SetColors(const std::vector<glm::vec3>& colors);
	void SetTexcoords(const std::vector<glm::vec2>& texcoords);
	void SetIndices(const std::vector<glm::ivec3>& indices);

	glm::vec3* d_positions;
	glm::vec3* d_colors;
	glm::vec3* d_normals;
	glm::vec2* d_texcoords;

	glm::ivec3* d_indices;

	glm::mat3 rotation;
	glm::vec3 translation;
	std::vector<glm::mat3> v_rotation;
	std::vector<glm::vec3> v_translation;
	int num_positions;
	int num_colors;
	int num_normals;
	int num_texcoords;
	int num_indices;
};

class FrameBuffer
{
public:
	FrameBuffer();
	FrameBuffer(int rows, int cols, float cx, float cy, float fx, float fy);
	~FrameBuffer();

	void Create(int rows, int cols, float cx, float cy, float fx, float fy, int _num_particles);
	void Initialize(int rows, int cols);
	void ClearBuffer();
	void Reset();

	//cv::Mat GetImage();
	void GetDepth(float* depth);
	void GetMatchingScore(float* score_buffer);
	void SetMatchingScore(float score);
	void SetSrcDepth(float* depth_src, bool* other_objects_regions);
	void SetParticleNum(int num_particles);
	void SetThreshold(float threshold);
	// void GetMatchingScore(float* depth, float* depth_src, uint8_t* other_objects_regions);
	void GetVMap(glm::ivec3* vindices, glm::vec3* vweights, int* findices);
	int* d_z;
	int* d_colors;
	int* d_findices;
	glm::vec3* d_vweights;
	glm::ivec3* d_vindices;
	glm::mat3* p_rotation;
	glm::vec3* p_translation;

	float* d_depth;
	int row, col;
	float cx, cy, fx, fy;

	float* depth_src_ptr;
	bool* other_objects_regions_ptr;
	int* score_map_ptr;
	float* matching_scores;
	int matching_score_index;
	int size_of_matching_scores_buffer;
	float c_threshold;
	int num_particles;

	unsigned int* score_count_ptr;
	unsigned int* inter_count_ptr;
	unsigned int* union_count_ptr;
};

template<class T>
void FreeCudaArray(T* &array, int& num) {
	if (array) {
		cudaFree(array);
		num = 0;
		array = 0;
	}
}

template<class T>
void FreeCudaImage(T* &array, int& row, int& col) {
	if (array) {
		cudaSetDevice(1);
		cudaFree(array);
		cudaSetDevice(0);
		row = 0;
		col = 0;
		array = 0;
	}
}

#endif