#include <iostream>
#include <cstring>
#include "buffer.hpp"
#include "render.hpp"

extern "C" {

std::vector<VertexBuffer*> vertexBuffers;
FrameBuffer frameBuffer;

void InitializeCamera(int width, int height, float fx, float fy, float cx, float cy, int num_particles)
{
	frameBuffer.Create(height, width, cx, cy, fx, fy, num_particles);
	SetCameraParam(frameBuffer);
}

int SetMesh(glm::vec3* positions, glm::ivec3* faces, int num_v, int num_f) {
	vertexBuffers.push_back(new VertexBuffer());
	std::vector<glm::vec3> p(num_v);
	std::vector<glm::ivec3> f(num_f);
	memcpy(p.data(), positions, sizeof(glm::vec3) * num_v);
	memcpy(f.data(), faces, sizeof(glm::ivec3) * num_f);

	vertexBuffers.back()->SetPositions(p);
	vertexBuffers.back()->SetIndices(f);

	return vertexBuffers.size() - 1;
}

void SetTransformSingle(int handle, float* transform) {
	glm::mat3 rotation;
	glm::vec3 translation;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			rotation[i][j] = transform[j * 4 + i];
		}
		translation[i] = transform[i * 4 + 3];
	}

	vertexBuffers[handle]->rotation = rotation;
	vertexBuffers[handle]->translation = translation;
}
void SetTransform(int handle, float* transform) {

	for(int k = 0; k < frameBuffer.num_particles; ++k) {
		glm::mat3 rotation;
		glm::vec3 translation;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				rotation[i][j] = transform[k * 16 + j * 4 + i];
			}
			translation[i] = transform[k * 16 + i * 4 + 3];
		}
		vertexBuffers[handle]->v_rotation.push_back(rotation);
		vertexBuffers[handle]->v_translation.push_back(translation);
	}
}

void ClearData() {
	for (auto vb : vertexBuffers)
		delete vb;
	vertexBuffers.clear();
}

void Render(int handle) {
	frameBuffer.ClearBuffer();
	
	cudaMemcpy(frameBuffer.p_rotation, vertexBuffers[handle]->v_rotation.data(), sizeof(glm::mat3) * frameBuffer.num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(frameBuffer.p_translation, vertexBuffers[handle]->v_translation.data(), sizeof(glm::vec3) * frameBuffer.num_particles, cudaMemcpyHostToDevice);

	Render(*vertexBuffers[handle], frameBuffer);
	vertexBuffers[handle]->v_rotation.clear();
	vertexBuffers[handle]->v_translation.clear();
}

void Render_once(int handle) {
	frameBuffer.ClearBuffer();
	Render_once(*vertexBuffers[handle], frameBuffer);
}


void GetDepth(float* depth) {
	FetchDepth(frameBuffer);
	frameBuffer.GetDepth(depth);
}


void SetNumOfParticles(int num_particles, int threshold) {
	frameBuffer.SetParticleNum(num_particles);
	frameBuffer.SetThreshold(float(float(threshold) * 0.000001));
}
void SetSrcDepth(float* depth_src, bool* other_objects_regions) {
	frameBuffer.SetSrcDepth(depth_src, other_objects_regions);
}

void GetMatchingScores(float* score_buffer) {
	CalcMatchingScore_GPU(frameBuffer);
	frameBuffer.GetMatchingScore(score_buffer);
}

void GetVMap(int handle, glm::ivec3* vindices, glm::vec3* vweights, int* findices) {
	FetchVMap(*vertexBuffers[handle], frameBuffer);
	frameBuffer.GetVMap(vindices, vweights, findices);
}

void Colorize(glm::vec4* VC, glm::ivec3* vindices, glm::vec3* vweights, unsigned char* mask, glm::vec3* image, int row, int col) {
	int offset = 0;
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			if (mask[offset]) {
				glm::vec3& c = image[offset];
				for (int k = 0; k < 3; ++k) {
					float w = vweights[offset][k];
					int index = vindices[offset][k];
					glm::vec4 color(c * w, w);
					VC[index] += color;
				}
			}
			offset += 1;
		}
	}
}

};
