#pragma once

#include "globalstate.h"

#include "opencv2/core/core.hpp"

struct Camera {
    Camera () : P ( cv::Mat::eye ( 3,4,CV_32F ) ),  R ( cv::Mat::eye ( 3,3,CV_32F ) ),baseline (0.54f), reference ( false ), depthMin ( 2.0f ), depthMax ( 20.0f ) {}
    cv::Mat_<float> P;
    cv::Mat_<float> P_inv;
    cv::Mat_<float> M_inv;
    //cv::Mat_<float> K;
    cv::Mat_<float> R;
    cv::Mat_<float> R_orig_inv;
    cv::Mat_<float> t;
    cv::Vec3f C;
    float baseline;
    bool reference;
    float depthMin; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    float depthMax; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    //int id; //corresponds to the image name id (eg. 0-10), independent of order in argument list, just dependent on name
    std::string id;
    cv::Mat_<float> K;
    cv::Mat_<float> K_inv;
    //float f;
};

//parameters for camera geometry setup (assuming that K1 = K2 = K, P1 = K [I | 0] and P2 = K [R | t])
struct CameraParameters {
    CameraParameters () : rectified ( false ), idRef ( 0 ) {}
    cv::Mat_<float> K; //if K varies from camera to camera: K and f need to be stored within Camera
    cv::Mat_<float> K_inv; //if K varies from camera to camera: K and f need to be stored within Camera
    float f;
    bool rectified;
    std::vector<Camera> cameras;
    int idRef;
    std::vector<int> viewSelectionSubset;
};

int runcuda(GlobalState &gs);

void delTexture (int num, cudaTextureObject_t texs[], cudaArray *cuArray[]);

void addImageToTextureUint(std::vector<cv::Mat_<uint8_t> > &imgs,
                           cudaTextureObject_t texs[], cudaArray *cuArray[]);

void addImageToTextureFloatColor(std::vector<cv::Mat> &imgs,
                                 cudaTextureObject_t texs[],
                                 cudaArray *cuArray[]);

void addImageToTextureFloatGray(std::vector<cv::Mat> &imgs,
                                cudaTextureObject_t texs[],
                                cudaArray *cuArray[]);
