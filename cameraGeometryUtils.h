/*
 *  cameraGeometryUtils.h
 *
 * utility functions for camera geometry related stuff
 * most of them from: "Multiple View Geometry in computer vision" by Hartley and Zisserman
 */

#pragma once
#include "files.h"
#include "fileIoUtils.h"
#include "mathUtils.h"
#include "gipuma.h"
#include <limits>
#include <signal.h>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/core/utility.hpp"
#endif

cv::Mat_<float> getColSubMat ( cv::Mat_<float> M, int* indices, int numCols ) {
    cv::Mat_<float> subMat = cv::Mat::zeros ( M.rows,numCols,CV_32F );
    for ( int i = 0; i < numCols; i++ ) {
        M.col ( indices[i] ).copyTo ( subMat.col ( i ) );
    }
    return subMat;
}

// Multi View Geometry, page 163
cv::Mat_<float> getCameraCenter ( cv::Mat_<float> &P ) {
    cv::Mat_<float> C = cv::Mat::zeros ( 4,1,CV_32F );

    cv::Mat_<float> M = cv::Mat::zeros ( 3,3,CV_32F );

    int xIndices[] = { 1, 2, 3 };
    int yIndices[] = { 0, 2, 3 };
    int zIndices[] = { 0, 1, 3 };
    int tIndices[] = { 0, 1, 2 };

    // x coordinate
    M = getColSubMat ( P,xIndices,sizeof ( xIndices )/sizeof ( xIndices[0] ) );
    C ( 0,0 ) = ( float )determinant ( M );

    // y coordinate
    M = getColSubMat ( P,yIndices,sizeof ( yIndices )/sizeof ( yIndices[0] ) );
    C ( 1,0 ) = - ( float )determinant ( M );

    // z coordinate
    M = getColSubMat ( P,zIndices,sizeof ( zIndices )/sizeof ( zIndices[0] ) );
    C ( 2,0 ) = ( float )determinant ( M );

    // t coordinate
    M = getColSubMat ( P,tIndices,sizeof ( tIndices )/sizeof ( tIndices[0] ) );
    C ( 3,0 ) = - ( float )determinant ( M );

    return C;
}

inline cv::Vec3f get3Dpoint ( Camera &cam, float x, float y, float depth ) {
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first

    cv::Mat_<float> pt = cv::Mat::ones ( 3,1,CV_32F );
    pt ( 0,0 ) = x;
    pt ( 1,0 ) = y;

    //formula taken from page 162 (alternative expression)
    cv::Mat_<float> ptX = cam.M_inv * ( depth*pt - cam.P.col ( 3 ) );
    return cv::Vec3f ( ptX ( 0 ),ptX ( 1 ),ptX ( 2 ) );
}

inline cv::Vec3f get3Dpoint ( Camera &cam, int x, int y, float depth ){
    return get3Dpoint(cam,(float)x,(float)y,depth);
}

// get the viewing ray for a pixel position of the camera
static inline cv::Vec3f getViewVector ( Camera &cam, int x, int y) {

    //get some point on the line (the other point on the line is the camera center)
    cv::Vec3f ptX = get3Dpoint ( cam,x,y,1.0f );

    //get vector between camera center and other point on the line
    cv::Vec3f v = ptX - cam.C;
    return normalize ( v );
}

/* get depth from 3D point
 * page 162: w = P3T*X (P3T ... third (=last) row of projection matrix P)
 */
float getDepth ( cv::Vec3f &X, cv::Mat_<float> &P ) {
    //assuming homogenous component of X being 1
    float w =  P ( 2,0 )*X ( 0 ) + P ( 2,1 ) * X ( 1 ) + P ( 2,2 ) * X ( 2 ) + P ( 2,3 );

    return w;
}

cv::Mat_<float> getTransformationMatrix ( cv::Mat_<float> R, cv::Mat_<float> t ) {
    cv::Mat_<float> transMat = cv::Mat::eye ( 4,4, CV_32F );
    //Mat_<float> Rt = - R * t;
    R.copyTo ( transMat ( cv::Range ( 0,3 ),cv::Range ( 0,3 ) ) );
    t.copyTo ( transMat ( cv::Range ( 0,3 ),cv::Range ( 3,4 ) ) );

    return transMat;
}

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
float disparityDepthConversion ( float f, float baseline, float d ) {
    /*if ( d == 0 )*/
        /*return FLT_MAX;*/
    return f * baseline / d;
}

cv::Mat_<float> getTransformationReferenceToOrigin ( cv::Mat_<float> R,cv::Mat_<float> t ) {
    // create rotation translation matrix
    cv::Mat_<float> transMat_original = getTransformationMatrix ( R,t );

    // get transformation matrix for [R1|t1] = [I|0]
    return transMat_original.inv ();
}

void transformCamera ( cv::Mat_<float> R,cv::Mat_<float> t, cv::Mat_<float> transform, Camera &cam, cv::Mat_<float> K ) {
    // create rotation translation matrix
    cv::Mat_<float> transMat_original = getTransformationMatrix ( R,t );

    //transform
    cv::Mat_<float> transMat_t = transMat_original * transform;

    // compute translated P (only consider upper 3x4 matrix)
    cam.P = K * transMat_t ( cv::Range ( 0,3 ),cv::Range ( 0,4 ) );
    // set R and t
    cam.R = transMat_t ( cv::Range ( 0,3 ),cv::Range ( 0,3 ) );
    cam.t = transMat_t ( cv::Range ( 0,3 ),cv::Range ( 3,4 ) );
    // set camera center C
    cv::Mat_<float> C = getCameraCenter ( cam.P );

    C = C / C ( 3,0 );
    cam.C = cv::Vec3f ( C ( 0,0 ),C ( 1,0 ),C ( 2,0 ) );
}

cv::Mat_<float> scaleK ( cv::Mat_<float> K, float scaleFactor ) {

    cv::Mat_<float> K_scaled = K.clone();
    //scale focal length
    K_scaled ( 0,0 ) = K ( 0,0 ) / scaleFactor;
    K_scaled ( 1,1 ) = K ( 1,1 ) / scaleFactor;
    //scale center point
    K_scaled ( 0,2 ) = K ( 0,2 ) / scaleFactor;
    K_scaled ( 1,2 ) = K ( 1,2 ) / scaleFactor;

    return K_scaled;
}
void copyOpencvVecToFloat4 ( cv::Vec3f &v, float4 *a)
{
    a->x = v(0);
    a->y = v(1);
    a->z = v(2);
}
void copyOpencvVecToFloatArray ( cv::Vec3f &v, float *a)
{
    a[0] = v(0);
    a[1] = v(1);
    a[2] = v(2);
}
void copyOpencvMatToFloatArray ( cv::Mat_<float> &m, float **a)
{
    for (int pj=0; pj<m.rows ; pj++)
        for (int pi=0; pi<m.cols ; pi++)
        {
            (*a)[pi+pj*m.cols] = m(pj,pi);
        }
}


CameraParameters getCameraParameters ( CameraParameters_cu &cpc,
                                       CameraParameters& params,
                                       float scaleFactor,
                                       bool transformP);
/* get camera parameters (e.g. projection matrices) from file
 * Input:  inputFiles  - pathes to calibration files
 *         scaleFactor - if image was rescaled we need to adapt calibration matrix K accordingly
 * Output: camera parameters
 */
CameraParameters getCameraParameters ( CameraParameters_cu &cpc,
                                       InputFiles inputFiles,
                                       float scaleFactor = 1.0f,
                                       bool transformP = true )
{

    CameraParameters params;
    size_t numCameras = 2;
    params.cameras.resize ( numCameras );
    //get projection matrices

    //load projection matrix from file (e.g. for Kitti)
    if ( !inputFiles.calib_filename.empty () ) {
        //two view case
        readCalibFileKitti ( inputFiles.calib_filename,params.cameras[0].P,params.cameras[1].P );
        params.rectified = false; // for Kitti data is actually rectified, set this to true for computation in disparity space

    }
    cv::Mat_<float> KMaros = cv::Mat::eye ( 3, 3, CV_32F );
    KMaros(0,0) = 8066.0;
    KMaros(1,1) = 8066.0;
    KMaros(0,2) = 2807.5;
    KMaros(1,2) = 1871.5;
    //load projection matrix from file (e.g. for Strecha)

    // Load pmvs files
    if ( !inputFiles.pmvs_folder.empty () ) {
        numCameras = inputFiles.img_filenames.size ();
        params.cameras.resize ( numCameras );
        for ( size_t i = 0; i < numCameras; i++ ) {
            int lastindex = inputFiles.img_filenames[i].find_last_of(".");
            std::string filename_without_extension = inputFiles.img_filenames[i].substr(0, lastindex);
            readPFileStrechaPmvs ( inputFiles.p_folder + filename_without_extension + ".txt",params.cameras[i].P );
            unsigned found = inputFiles.img_filenames[i].find_last_of ( "." );
            //params.cameras[i].id = atoi ( inputFiles.img_filenames[i].substr ( 0,found ).c_str () );
            params.cameras[i].id = inputFiles.img_filenames[i].substr ( 0,found ).c_str ();
            // params.cameras[i].P = KMaros * params.cameras[i].P;
            //cout << params.cameras[i].P << endl;

        }
    }
    // Load p files strecha style
    else if ( !inputFiles.p_folder.empty () ) {
        numCameras = inputFiles.img_filenames.size ();
        params.cameras.resize ( numCameras );
        for ( size_t i = 0; i < numCameras; i++ ) {
            readPFileStrechaPmvs ( inputFiles.p_folder + inputFiles.img_filenames[i] + ".P",params.cameras[i].P );
            unsigned found = inputFiles.img_filenames[i].find_last_of ( "." );
            //params.cameras[i].id = atoi ( inputFiles.img_filenames[i].substr ( 0,found ).c_str () );
            params.cameras[i].id = inputFiles.img_filenames[i].substr ( 0,found ).c_str ();
           // params.cameras[i].P = KMaros * params.cameras[i].P;

            //cout << params.cameras[i].P << endl;
        }

    }
    // Load P matrix for middlebury format
    if ( !inputFiles.krt_file.empty () ) {
        numCameras = inputFiles.img_filenames.size ();
        params.cameras.resize ( numCameras );
        //cout << "Num Cameras " << numCameras << endl;
        readKRtFileMiddlebury ( inputFiles.krt_file, params.cameras, inputFiles);
    }

    /*cout << "KMaros is" << endl;*/
    /*cout << KMaros << endl;*/

    return getCameraParameters(cpc, params, scaleFactor, transformP);
}

CameraParameters getCameraParameters(CameraParameters_cu &cpc,
                                     CameraParameters& params,
                                     float scaleFactor = 1.0f,
                                     bool transformP = true) {
    size_t numCameras = params.cameras.size();

    // decompose projection matrices into K, R and t
    std::vector<cv::Mat_<float> > K ( numCameras );
    std::vector<cv::Mat_<float> > R ( numCameras );
    std::vector<cv::Mat_<float> > T ( numCameras );

    std::vector<cv::Mat_<float> > C ( numCameras );
    std::vector<cv::Mat_<float> > t ( numCameras );

    for ( size_t i = 0; i < numCameras; i++ ) {
        decomposeProjectionMatrix ( params.cameras[i].P,K[i],R[i],T[i] );

        //cout << "K: " << K[i] << endl;
        //cout << "R: " << R[i] << endl;
        //cout << "T: " << T[i] << endl;

        // get 3-dimensional translation vectors and camera center (divide by augmented component)
        C[i] = T[i] ( cv::Range ( 0,3 ),cv::Range ( 0,1 ) ) / T[i] ( 3,0 );
        t[i] = -R[i] * C[i];

        //cout << "C: " << C[i] << endl;
        //cout << "t: " << t[i] << endl;
    }

    // transform projection matrices (R and t part) so that P1 = K [I | 0]
    //computeTranslatedProjectionMatrices(R1, R2, t1, t2, params);
    cv::Mat_<float> transform = cv::Mat::eye ( 4,4 ,CV_32F);

    if ( transformP )
        transform = getTransformationReferenceToOrigin ( R[0],t[0] );
    /*cout << "transform is " << transform << endl;*/
    params.cameras[0].reference = true;
    params.idRef = 0;
    //cout << "K before scale is" << endl;
    //cout << K[0] << endl;

    //assuming K is the same for all cameras
    params.K = scaleK ( K[0],scaleFactor );
    params.K_inv = params.K.inv ();
    // get focal length from calibration matrix
    params.f = params.K ( 0,0 );

    for ( size_t i = 0; i < numCameras; i++ ) {
        params.cameras[i].K = scaleK(K[i],scaleFactor);
        params.cameras[i].K_inv = params.cameras[i].K.inv ( );
        //params.cameras[i].f = params.cameras[i].K(0,0);

        // if ( !inputFiles.bounding_folder.empty () ) {
        //     cv::Vec3f ptBL, ptTR;
        //     readBoundingVolume ( inputFiles.bounding_folder + inputFiles.img_filenames[i] + ".bounding",ptBL,ptTR );

        //     cout << "d1: " << getDepth ( ptBL,params.cameras[i].P ) <<endl;
        //     cout << "d2: " << getDepth ( ptTR,params.cameras[i].P ) <<endl;
        // }

        params.cameras[i].R_orig_inv = R[i].inv(cv::DECOMP_SVD);
        transformCamera ( R[i],t[i], transform,    params.cameras[i],params.K );

        params.cameras[i].P_inv = params.cameras[i].P.inv ( cv::DECOMP_SVD );
        params.cameras[i].M_inv = params.cameras[i].P.colRange ( 0,3 ).inv ();

        // set camera baseline (if unknown we need to guess something)
        //float b = (float)norm(t1,t2,NORM_L2);
        params.cameras[i].baseline = 0.54f; //0.54 = Kitti baseline

        // K
        cv::Mat_<float> tmpK = params.K.t ();
        //copyOpencvMatToFloatArray ( params.K, &cpc.K);
        //copyOpencvMatToFloatArray ( params.K_inv, &cpc.K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].K, &cpc.cameras[i].K);
        copyOpencvMatToFloatArray ( params.cameras[i].K_inv, &cpc.cameras[i].K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].R_orig_inv, &cpc.cameras[i].R_orig_inv);
        cpc.cameras[i].fy = params.K(1,1);
        cpc.f = params.K(0,0);
        cpc.cameras[i].f = params.K(0,0);
        cpc.cameras[i].fx = params.K(0,0);
        cpc.cameras[i].fy = params.K(1,1);
        cpc.cameras[i].baseline = params.cameras[i].baseline;
        cpc.cameras[i].reference = params.cameras[i].reference;

        /*params.cameras[i].alpha = params.K ( 0,0 )/params.K(1,1);*/
        cpc.cameras[i].alpha = params.K ( 0,0 )/params.K(1,1);
        // Copy data to cuda structure
        copyOpencvMatToFloatArray ( params.cameras[i].P,     &cpc.cameras[i].P);
        copyOpencvMatToFloatArray ( params.cameras[i].P_inv, &cpc.cameras[i].P_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].M_inv, &cpc.cameras[i].M_inv);
        //copyOpencvMatToFloatArray ( params.K,                &cpc.cameras[i].K);
        //copyOpencvMatToFloatArray ( params.K_inv,            &cpc.cameras[i].K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].K,                &cpc.cameras[i].K);
        copyOpencvMatToFloatArray ( params.cameras[i].K_inv,            &cpc.cameras[i].K_inv);
        copyOpencvMatToFloatArray ( params.cameras[i].R,     &cpc.cameras[i].R);
        /*copyOpencvMatToFloatArray ( params.cameras[i].t, &cpc.cameras[i].t);*/
        /*copyOpencvVecToFloatArray ( params.cameras[i].C, cpc.cameras[i].C);*/
        copyOpencvVecToFloat4 ( params.cameras[i].C,         &cpc.cameras[i].C4);
        cpc.cameras[i].t4.x = params.cameras[i].t(0);
        cpc.cameras[i].t4.y = params.cameras[i].t(1);
        cpc.cameras[i].t4.z = params.cameras[i].t(2);
        cv::Mat_<float> tmp = params.cameras[i].P.col(3);
        /*cpc.cameras[i].P_col3[0] = tmp(0,0);*/
        /*cpc.cameras[i].P_col3[1] = tmp(1,0);*/
        /*cpc.cameras[i].P_col3[2] = tmp(2,0);*/
        cpc.cameras[i].P_col34.x = tmp(0,0);
        cpc.cameras[i].P_col34.y = tmp(1,0);
        cpc.cameras[i].P_col34.z = tmp(2,0);
        //cout << params.cameras[i].P << endl;
        //cout << endl;

        cv::Mat_<float> tmpKinv = params.K_inv.t ();
    }

    return params;
}

void selectViews(CameraParameters &cameraParams, int imgWidth, int imgHeight,
                 AlgorithmParameters &algParams) {
  std::vector<Camera> &cameras = cameraParams.cameras;
    Camera ref = cameras[cameraParams.idRef];

    int x = imgWidth / 2;
    int y = imgHeight / 2;

    cameraParams.viewSelectionSubset.clear ();

    cv::Vec3f viewVectorRef = getViewVector ( ref, x, y);

    // TODO hardcoded value makes it a parameter
    float minimum_angle_degree = algParams.min_angle;
    float maximum_angle_degree = algParams.max_angle;

    unsigned int maximum_view = algParams.max_views;
    float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
    float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
    float min_depth = 9999;
    float max_depth = 0;
    if ( algParams.viewSelection )
        printf("Accepting intersection angle of central rays from %f to %f degrees, use --min_angle=<angle> and --max_angle=<angle> to modify them\n", minimum_angle_degree, maximum_angle_degree);
    for ( size_t i = 1; i < cameras.size (); i++ ) {
        //if ( !algParams.viewSelection ) { //select all views, dont perform selection
            //cameraParams.viewSelectionSubset.push_back ( i );
            //continue;
        //}

        cv::Vec3f vec = getViewVector(cameras[i], x, y);

        float baseline = cv::norm (cameras[0].C, cameras[i].C);
        float angle = getAngle ( viewVectorRef, vec );
        printf("baseline %f, angle %f\n", baseline, angle * 180.0f / M_PI);
        if ( angle > minimum_angle_radians &&
             angle < maximum_angle_radians ) //0.6 select if angle between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees suggested by some paper)
        {
            if ( algParams.viewSelection ) {
                cameraParams.viewSelectionSubset.push_back ( i );
                //printf("\taccepting camera %ld with angle\t %f degree (%f radians) and baseline %f\n", i, angle*180.0f/M_PI, angle, baseline);
            }
            float min_range = (baseline/2.0f) / sin(maximum_angle_radians/2.0f);
            float max_range = (baseline/2.0f) / sin(minimum_angle_radians/2.0f);
            min_depth = std::min(min_range, min_depth);
            max_depth = std::max(max_range, max_depth);
            //printf("Min max ranges are %f %f\n", min_range, max_range);
            //printf("Min max depth are %f %f\n", min_depth, max_depth);
        }
        //else
            //printf("Discarding camera %ld with angle\t %f degree (%f radians) and baseline, %f\n", i, angle*180.0f/M_PI, angle, baseline);
    }

    if (algParams.depthMin == -1)
        algParams.depthMin = min_depth;
    if (algParams.depthMax == -1)
        algParams.depthMax = max_depth;

    if (!algParams.viewSelection) {
        cameraParams.viewSelectionSubset.clear();
        for ( size_t i = 1; i < cameras.size (); i++ )
            cameraParams.viewSelectionSubset.push_back ( i );
        return;
    }
    if (cameraParams.viewSelectionSubset.size() >= maximum_view) {
        printf("Too many camera, randomly selecting only %d of them (modify with --max_views=<number>)\n", maximum_view);
        std::srand ( unsigned ( std::time(0) ) );
        std::random_shuffle( cameraParams.viewSelectionSubset.begin(), cameraParams.viewSelectionSubset.end() ); // shuffle elements of v
        cameraParams.viewSelectionSubset.erase (cameraParams.viewSelectionSubset.begin()+maximum_view,cameraParams.viewSelectionSubset.end());
    }
    //for (auto i : cameraParams.viewSelectionSubset )
        //printf("\taccepting camera %d\n", i);
}
