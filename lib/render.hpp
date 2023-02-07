#ifndef RENDER_H_
#define RENDER_H_

#include "buffer.hpp"
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/core/core.hpp>

void NaiveRender(FrameBuffer& frameBuffer);
void Render(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer, int renderPrimitive = 1);
void Render_once(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer, int renderPrimitive = 1);
void FetchDepth(FrameBuffer& frameBuffer);
void FetchVMap(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer);
void CalcMatchingScore_GPU(FrameBuffer& frameBuffer);
void SetCameraParam(FrameBuffer& frameBuffer);

#endif