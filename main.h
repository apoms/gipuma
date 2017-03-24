#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/core/utility.hpp"
#endif

#include <omp.h>
#include <stdint.h>

using namespace cv;
using namespace std;

typedef Vec<uint16_t, 2> Vec2us;

//parameters of algorithms
//struct AlgorithmParameters {
    //AlgorithmParameters () : algorithm ( PM_COST ), max_disparity ( 256.0f ), min_disparity ( 0.0f ), box_hsize ( 15 ), box_vsize ( 15 ), tau_color ( 10.0f ), tau_gradient ( 2.0f ), alpha ( 0.9f ), gamma ( 10.0f ), border_value ( -1 ), iterations ( 3 ), color_processing ( false ), dispTol ( 1.0f ), normTol ( 0.1f ), census_epsilon ( 2.5f ), self_similarity_n ( 50 ), cam_scale ( 1.0f ), num_img_processed ( 1 ), costThresh ( 40.0f ), n_best ( 2 ), viewSelection ( false ), good_factor ( 1.8f ), cost_comb ( COMB_BEST_N ), depthMin ( 2.0f ), depthMax ( 20.0f ) {}
    //int algorithm; // algorithm cost type
    //float max_disparity; // maximal disparity value
    //float min_disparity; // minimum disparity value (default 0)
    //int box_hsize; // filter kernel width
    //int box_vsize; // filter kernel height
    //float tau_color; // PM_COST max. threshold for color
    //float tau_gradient; // PM_COST max. threshold for gradient
    //float alpha; // PM_COST weighting between color and gradient
    //float gamma; // parameter for weight function (used e.g. in PM_COST)
    //int border_value; // what value should pixel at extended border get (constant or replicate -1)
    //int iterations; // number of iterations
    //bool color_processing; // use color processing or not (otherwise just grayscale processing)
    //float dispTol; //PM Stereo: 1, PM Huber: 0.5
    //float normTol; // 0.1 ... about 5.7 degrees
    //float census_epsilon; //for census transform
    //int self_similarity_n; // number of pixels considered for self similarity
    //float cam_scale; //used to rescale K in case of rescaled image size
    //int num_img_processed; //number of images that are processed as reference images
    //float costThresh; // threshold to decide whether disparity/depth is valid or not
    //float good_factor; // for cost aggregation/combination good: factor for truncation
    //int n_best;
    //int cost_comb;
    //bool viewSelection;
    //float depthMin;
    //float depthMax;
//};

struct Results {
    Results () : error_occ ( 1.0f ), error_noc ( 1.0f ), valid_pixels ( 0.0f ), error_valid ( 1.0f ), error_valid_all ( 1.0f ), total_runtime ( 0.0f ), runtime_per_pixel ( 0.0f ) {}
    float error_occ;
    float error_noc;
    float valid_pixels; // passed occlusion check
    float valid_pixels_gt;
    float error_valid;
    float error_valid_all;
    double total_runtime;
    double runtime_per_pixel;
};

struct Plane {
    Mat_<Vec3f> normal;
    Mat_<float> d;
    void release () {
        normal.release ();
        d.release ();
    }
};
