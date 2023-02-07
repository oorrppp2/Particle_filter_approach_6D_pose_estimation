from ctypes import *
import numpy as np
import os
import sys

libpath = os.path.dirname(os.path.abspath(__file__))
Render = cdll.LoadLibrary(libpath + '/libRender.so')


def setup(info):
  Render.InitializeCamera(info['Width'], info['Height'],
    c_float(info['fx']), c_float(info['fy']), c_float(info['cx']), c_float(info['cy']), info['num_particles'])

def SetMesh(V, F):
  handle = Render.SetMesh(c_void_p(V.ctypes.data), c_void_p(F.ctypes.data), V.shape[0], F.shape[0])
  return handle

def render(handle, world2cam):
  Render.SetTransform(handle, c_void_p(world2cam.ctypes.data))
  Render.Render(handle);

def render_once(handle, world2cam):
  Render.SetTransformSingle(handle, c_void_p(world2cam.ctypes.data))
  Render.Render_once(handle);

def getDepth(info):
  depth = np.zeros((info['num_particles'], info['Height'],info['Width']), dtype='float32')
  Render.GetDepth(c_void_p(depth.ctypes.data))
  return depth

""" Register the observed depth image into the renderer. """
def setSrcDepthImage(info, depth_src, other_objects_region):
  other_objects_region = np.asarray(other_objects_region, dtype=np.bool)
  depth_src = np.asarray(depth_src, dtype=np.float32)
  Render.SetSrcDepth(c_void_p(depth_src.ctypes.data), c_void_p(other_objects_region.ctypes.data))

def setNumOfParticles(num_particles, threshold):
  Render.SetNumOfParticles(c_void_p(num_particles), c_void_p(threshold))

def getMatchingScores(num_particles):
  matching_score_buffer = np.zeros((num_particles), dtype='float32')
  Render.GetMatchingScores(c_void_p(matching_score_buffer.ctypes.data))
  return matching_score_buffer

def getVMap(handle, info):
  vindices = np.zeros((info['Height'],info['Width'], 3), dtype='int32')
  vweights = np.zeros((info['Height'],info['Width'], 3), dtype='float32')
  findices = np.zeros((info['Height'], info['Width']), dtype='int32')

  Render.GetVMap(handle, c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data), c_void_p(findices.ctypes.data))

  return vindices, vweights, findices

def colorize(VC, vindices, vweights, mask, cimage):
  Render.Colorize(c_void_p(VC.ctypes.data), c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data),
    c_void_p(mask.ctypes.data), c_void_p(cimage.ctypes.data), vindices.shape[0], vindices.shape[1])
  
def Clear():
  Render.ClearData()
