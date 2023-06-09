/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"
#include "ImuTypes.h"


#include "GeometricCamera.h"
#include "SerializationUtils.h"

#include <mutex>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>


// Edge-SLAM
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/set.hpp>


// YoloSLAM
// added for the YoloV8 detector
#include "YoloV8.h"

namespace ORB_SLAM3
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class GeometricCamera;


class KeyFrame
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        //ar & nNextId;
        ar & mnId;
        ar & mnMapId;
        ar & mbInsertLoop;
        ar & const_cast<long unsigned int&>(mnFrameId);
        ar & const_cast<double&>(mTimeStamp);
        // Grid
        ar & const_cast<int&>(mnGridCols);
        ar & const_cast<int&>(mnGridRows);
        ar & const_cast<float&>(mfGridElementWidthInv);
        ar & const_cast<float&>(mfGridElementHeightInv);
        // Variables of tracking
        ar & mnTrackReferenceForFrame;
        ar & mnFuseTargetForKF;
        
        // Variables of local mapping
        ar & mnBALocalForKF;
        ar & mnBAFixedForKF;
        
        ar & mnNumberOfOpt;
        
        
        // Variables used by KeyFrameDatabase
        ar & mnLoopQuery;
        ar & mnLoopWords;
        //Comment for now
        //ar & mLoopScore;
        ar & mnRelocQuery;
        ar & mnRelocWords;
        //Comment for now
        //ar & mRelocScore;
        //Comment for now
        ar & mnMergeQuery;
        //Comment for now
        ar & mnMergeWords;
        //Comment for now
        //ar & mMergeScore;
        ar & mnPlaceRecognitionQuery;
        ar & mnPlaceRecognitionWords;
        ar & mPlaceRecognitionScore;
        ar & mbCurrentPlaceRecognition;

        //Comment for now
        
        // Variables of loop closing
        serializeSophusSE3(ar,mTcwGBA,version);
        serializeSophusSE3(ar,mTcwBefGBA,version);
        ar & boost::serialization::make_array(mVwbGBA.data(), mVwbGBA.size());
        ar & boost::serialization::make_array(mVwbBefGBA.data(), mVwbBefGBA.size());

        ar & mBiasGBA;
        
        
        ar & mnBAGlobalForKF;
        
        //Comment for now
        ar& mnDataset;
        
        
        //Comment for now
        // Variables of Merging
        //serializeSophusSE3(ar,mTcwMerge,version);
        // serializeSophusSE3(ar,mTcwBefMerge,version);
        //serializeSophusSE3(ar,mTwcBefMerge,version);
        
        //ar & boost::serialization::make_array(mVwbMerge.data(), mVwbMerge.size());
         
        //ar & boost::serialization::make_array(mVwbBefMerge.data(), mVwbBefMerge.size());

        //ar & mBiasMerge;
        ar & mnMergeCorrectedForKF;
        //ar & mnMergeForKF;
        //ar & mfScaleMerge;
        ar & mnBALocalForMerge;
        
        //Comment for now
        // Scale
        ar & mfScale;

        // Calibration parameters
        ar & const_cast<float&>(fx);
        ar & const_cast<float&>(fy);
        ar & const_cast<float&>(invfx);
        ar & const_cast<float&>(invfy);
        ar & const_cast<float&>(cx);
        ar & const_cast<float&>(cy);
        ar & const_cast<float&>(mbf);
        ar & const_cast<float&>(mb);
        ar & const_cast<float&>(mThDepth);
        serializeMatrix(ar, mDistCoef, version);
        
        // Number of Keypoints
        ar & const_cast<int&>(N);
        

        // KeyPoints
        serializeVectorKeyPoints<Archive>(ar, mvKeys, version);
        serializeVectorKeyPoints<Archive>(ar, mvKeysUn, version);
        ar & const_cast<vector<float>& >(mvuRight);
        ar & const_cast<vector<float>& >(mvDepth);
        serializeMatrix<Archive>(ar,mDescriptors,version);
        
        // BOW
        ar & mBowVec;
        ar & mFeatVec;
        
        // Pose relative to parent
        serializeSophusSE3<Archive>(ar, mTcp, version);
        
                // Scale
        ar & const_cast<int&>(mnScaleLevels);
        ar & const_cast<float&>(mfScaleFactor);
        ar & const_cast<float&>(mfLogScaleFactor);
        ar & const_cast<vector<float>& >(mvScaleFactors);
        ar & const_cast<vector<float>& >(mvLevelSigma2);
        ar & const_cast<vector<float>& >(mvInvLevelSigma2);
        // Image bounds and calibration
        ar & const_cast<int&>(mnMinX);
        ar & const_cast<int&>(mnMinY);
        ar & const_cast<int&>(mnMaxX);
        ar & const_cast<int&>(mnMaxY);
        

        ///////////Pointer, so send the backup Id////
        ar & mBackupImuPreintegrated;
        ar & mBackupPrevKFId;
        ar & mBackupNextKFId;
        

        ar & mImuCalib;
                
        ar & mnOriginMapId;  
        ar & mNameFile;
        
        // sophus poses (already in SetPose())
        serializeSophusSE3<Archive>(ar, mTcw, version);
        //serializeSophusSE3<Archive>(ar, mTwc, version);
        
        //ar & boost::serialization::make_array(mRcw.data(), mRcw.size());
        //ar & boost::serialization::make_array(mRwc.data(), mRwc.size());
        
         //Velocity comment for now                     
         ar & boost::serialization::make_array(mVw.data(), mVw.size());
        ar & boost::serialization::make_array(mOwb.data(), mOwb.size());
        ar & mbHasVelocity;
        
        serializeSophusSE3<Archive>(ar, mTlr, version);
        serializeSophusSE3<Archive>(ar, mTrl, version);

        
        // Inertial variables
        ar & mImuBias;
        //Comment for now.
        ar & mvpMapPoints;
        
        // Grid
        ar & mGrid;
        // Connected KeyFrameWeight
        ar & mBackupConnectedKeyFrameIdWeights;
        
        /////////////mvpOrderedConnectedKeyFrames,mvOrderedWeights not saved
        

        // Spanning Tree and Loop Edges
        ar & mbFirstConnection;
        ar & mBackupParentId;
        ar & mvBackupChildrensId;
        ar & mvBackupLoopEdgesId;
        ar & mvBackupMergeEdgesId;
        

        // Bad flags
        ar & mbNotErase;
        ar & mbToBeErased;
        ar & mbBad;

        ar & mHalfBaseline;
        

        // Camera variables
        ar & mnBackupIdCamera;
        ar & mnBackupIdCamera2;
     
        ar & boost::serialization::make_array(mK_.data(), mK_.size());
        
                        
        // Pose

        // MapPointsId associated to keypoints
        ar & mvBackupMapPointsId;
        
        // Fisheye variables
        ar & mvLeftToRightMatch;
        ar & mvRightToLeftMatch;
        ar & const_cast<int&>(NLeft);
        ar & const_cast<int&>(NRight);
        serializeVectorKeyPoints<Archive>(ar, mvKeysRight, version);
        
        ar & mGridRight;

        // YoloSLAM: Method 1
        // Pixel values
        // size_t elem_size = mImg.elemSize();
        // size_t elem_type = mImg.type();
    
        // ar & mImg.cols;
        // ar & mImg.rows;
        // ar & elem_size;
        // ar & elem_type;

        // const size_t data_size = mImg.cols * mImg.rows * elem_size;
        // ar & boost::serialization::make_array(mImg.ptr(), data_size);
        //serializeMatrix<Archive>(ar,mImg,version);

        // YoloSLAM: Method 2
        ar & const_cast<vector<unsigned char>& >(encoded_img_data);
        

    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame();
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // YoloSLAM: Method 1
    //KeyFrame(Frame &F, cv::Mat im, Map* pMap, KeyFrameDatabase* pKFDB);
    // YoloSLAM: Method 2
    KeyFrame(Frame &F, std::vector<unsigned char> &encoded_img, Map* pMap, KeyFrameDatabase* pKFDB);


    // Pose functions
    void SetPose(const Sophus::SE3f &Tcw);
    void SetVelocity(const Eigen::Vector3f &Vw_);

    Sophus::SE3f GetPose();

    Sophus::SE3f GetPoseInverse();
    Eigen::Vector3f GetCameraCenter();

    Eigen::Vector3f GetImuPosition();
    Eigen::Matrix3f GetImuRotation();
    Sophus::SE3f GetImuPose();
    Eigen::Matrix3f GetRotation();
    Eigen::Vector3f GetTranslation();
    Eigen::Vector3f GetVelocity();
    bool isVelocitySet();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);

    void UpdateConnections(bool upParent=true);
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);
    void SetFirstConnection(bool bFirst);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // Merge Edges
    void AddMergeEdge(KeyFrame* pKF);
    set<KeyFrame*> GetMergeEdges();

    // MapPoint observation functions
    int GetNumberMPs();
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const int &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const int &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const bool bRight = false) const;
    bool UnprojectStereo(int i, Eigen::Vector3f &x3D);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }

    Map* GetMap();
    void UpdateMap(Map* pMap);

    void SetNewBias(const IMU::Bias &b);
    Eigen::Vector3f GetGyroBias();

    Eigen::Vector3f GetAccBias();

    IMU::Bias GetImuBias();

    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);
    bool ProjectPointUnDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    void PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam);
    void PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId);

    // YoloSLAM
    // void PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int,
    //               MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId, Inference &yolov8);


    void SetORBVocabulary(ORBVocabulary* pORBVoc);
    void SetKeyFrameDatabase(KeyFrameDatabase* pKFDB);

    bool bImu;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;
    
    ////////////////CommSLAM////////////////////////////
    long unsigned int mnBAIsAddedBefore;
    
    long unsigned int mnMapId;
    int isStartMap;
    
    long unsigned int mnEdgeForKF;
    long unsigned int mnEdgeFixedForKF;
    //Number of optimizations by BA(amount of iterations in BA)
    long unsigned int mnNumberOfOpt;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;
    long unsigned int mnMergeQuery;
    int mnMergeWords;
    float mMergeScore;
    long unsigned int mnPlaceRecognitionQuery;
    int mnPlaceRecognitionWords;
    float mPlaceRecognitionScore;

    bool mbCurrentPlaceRecognition;


    // Variables used by loop closing
    Sophus::SE3f mTcwGBA;
    Sophus::SE3f mTcwBefGBA;
    Eigen::Vector3f mVwbGBA;
    Eigen::Vector3f mVwbBefGBA;
    IMU::Bias mBiasGBA;
    long unsigned int mnBAGlobalForKF;

    // Variables used by merging
    Sophus::SE3f mTcwMerge;
    Sophus::SE3f mTcwBefMerge;
    Sophus::SE3f mTwcBefMerge;
    Eigen::Vector3f mVwbMerge;
    Eigen::Vector3f mVwbBefMerge;
    IMU::Bias mBiasMerge;
    long unsigned int mnMergeCorrectedForKF;
    long unsigned int mnMergeForKF;
    float mfScaleMerge;
    long unsigned int mnBALocalForMerge;

    float mfScale;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
    cv::Mat mDistCoef;
    

    
    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    Sophus::SE3f mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;

    // Preintegrated IMU measurements from previous keyframe
    KeyFrame* mPrevKF;
    KeyFrame* mNextKF;

    IMU::Preintegrated* mpImuPreintegrated;
    IMU::Calib mImuCalib;

    unsigned int mnOriginMapId;

    string mNameFile;

    int mnDataset;

    std::vector <KeyFrame*> mvpLoopCandKFs;
    std::vector <KeyFrame*> mvpMergeCandKFs;
    
    /////////////////CommSLAM//////////////////////////
    bool mbInsertLoop;
    //bool mbHasHessian;
    //cv::Mat mHessianPose;

    // YoloSLAM: Method 1
    //cv::Mat mImg;

    // YoloSLAM: Method 2
    std::vector<unsigned char> encoded_img_data;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:
    // sophus poses
    Sophus::SE3<float> mTcw;
    Eigen::Matrix3f mRcw;
    Sophus::SE3<float> mTwc;
    Eigen::Matrix3f mRwc;

    // IMU position
    Eigen::Vector3f mOwb;
    // Velocity (Only used for inertial SLAM)
    Eigen::Vector3f mVw;
    bool mbHasVelocity;

    //Transformation matrix between cameras in stereo fisheye
    Sophus::SE3<float> mTlr;
    Sophus::SE3<float> mTrl;

    // Imu bias
    IMU::Bias mImuBias;

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;
    // For save relation without pointer, this is necessary for save/load function
    std::vector<long long int> mvBackupMapPointsId;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;
    // For save relation without pointer, this is necessary for save/load function
    std::map<long unsigned int, int> mBackupConnectedKeyFrameIdWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;
    std::set<KeyFrame*> mspMergeEdges;
    // For save relation without pointer, this is necessary for save/load function
    long long int mBackupParentId;
    std::vector<long unsigned int> mvBackupChildrensId;
    std::vector<long unsigned int> mvBackupLoopEdgesId;
    std::vector<long unsigned int> mvBackupMergeEdgesId;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; // Only for visualization

    Map* mpMap;

    // Backup variables for inertial
    long long int mBackupPrevKFId;
    long long int mBackupNextKFId;
    IMU::Preintegrated mBackupImuPreintegrated;

    // Backup for Cameras
    unsigned int mnBackupIdCamera, mnBackupIdCamera2;

    // Calibration
    Eigen::Matrix3f mK_;

    // Mutex
    std::mutex mMutexPose; // for pose, velocity and biases
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
    std::mutex mMutexMap;

public:
    GeometricCamera* mpCamera, *mpCamera2;

    //Indexes of stereo observations correspondences
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    Sophus::SE3f GetRelativePoseTrl();
    Sophus::SE3f GetRelativePoseTlr();

    //KeyPoints in the right image (for stereo fisheye, coordinates are needed)
    const std::vector<cv::KeyPoint> mvKeysRight;

    const int NLeft, NRight;

    std::vector< std::vector <std::vector<size_t> > > mGridRight;

    Sophus::SE3<float> GetRightPose();
    Sophus::SE3<float> GetRightPoseInverse();

    Eigen::Vector3f GetRightCameraCenter();
    Eigen::Matrix<float,3,3> GetRightRotation();
    Eigen::Vector3f GetRightTranslation();
    
    
    void PrintPointDistribution(){
        int left = 0, right = 0;
        int Nlim = (NLeft != -1) ? NLeft : N;
        for(int i = 0; i < N; i++){
            if(mvpMapPoints[i]){
                if(i < Nlim) left++;
                else right++;
            }
        }
        cout << "Point distribution in KeyFrame: left-> " << left << " --- right-> " << right << endl;
    }


};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
