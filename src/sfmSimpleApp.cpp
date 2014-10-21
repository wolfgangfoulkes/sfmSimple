#include <vector>
#include <iostream>
#include <list>
#include <set>

#include "cinder/app/AppNative.h"
#include "cinder/gl/gl.h"
#include "cinder/ImageIo.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"
#include "cinder/qtime/QuickTime.h"

#include "opencv2/highgui/highgui.hpp"

#include "CinderOpenCv.h"

using namespace std;
using namespace ci;
using namespace ci::app;
using namespace cv;

#define intrpmnmx(val,min,max) (max==min ? 0.0 : ((val)-min)/(max-min))

struct CloudPoint {
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
};

class sfmSimpleApp : public AppNative {
  public:
	void setup();
    void update();
	void draw();
	
    VideoCapture capture;
	gl::Texture	mTexture;
    
    cv::Mat mFrame0;
    cv::Mat mFrame1;
    
    Matx34d P,P1;
    
    /*****Distance.h*****/
    std::vector<cv::KeyPoint> imgpts1;
    std::vector<cv::KeyPoint> imgpts2;
	std::vector<cv::KeyPoint> fullpts1;
    std::vector<cv::KeyPoint> fullpts2;
	std::vector<cv::KeyPoint> imgpts1_good;
    std::vector<cv::KeyPoint> imgpts2_good;

    //map of 2D points to Matches
	std::map<std::pair<int,int> ,std::vector<cv::DMatch> > matches_matrix;
	
    //computed P matrices for each view
	std::map<int,cv::Matx34d> Pmats;

    //from calibration
	cv::Mat K;
	cv::Mat_<double> Kinv;
	cv::Mat cam_matrix;
    cv::Mat distortion_coeff;

	std::vector<CloudPoint> pcloud;
	std::vector<cv::Vec3b> pointCloudRGB;
	std::vector<cv::KeyPoint> correspImg1Pt; //TODO: remove
	
	bool features_matched;
    
    void GetPosition(const Mat& frame1, const Mat& frame2);
    
    /*****Common.h*****/
    std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches);
    void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);
    void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps);

    std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);

    void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
                                   const std::vector<cv::KeyPoint>& imgpts2,
                                   const std::vector<cv::DMatch>& matches,
                                   std::vector<cv::KeyPoint>& pt_set1,
                                   std::vector<cv::KeyPoint>& pt_set2);
    
    /*****FindCameraMatrices.h*****/
    bool CheckCoherentRotation(cv::Mat_<double>& R);
    bool TestTriangulation(const std::vector<CloudPoint>& pcloud, const cv::Matx34d& P, std::vector<uchar>& status);
    
    void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w);
    bool DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2);
    
    cv::Mat GetFundamentalMat(	const std::vector<cv::KeyPoint>& imgpts1,
                                const std::vector<cv::KeyPoint>& imgpts2,
                                std::vector<cv::KeyPoint>& imgpts1_good,
                                std::vector<cv::KeyPoint>& imgpts2_good,
                                std::vector<cv::DMatch>& matches);
    
    bool FindCameraMatrices(const cv::Mat& K, 
                            const cv::Mat& Kinv,
                            const cv::Mat& distcoeff,
                            const std::vector<cv::KeyPoint>& imgpts1,   //from imgpoints
                            const std::vector<cv::KeyPoint>& imgpts2,   //from OpticalFlow
                            std::vector<cv::KeyPoint>& imgpts1_good,    //from imgpoints_good
                            std::vector<cv::KeyPoint>& imgpts2_good,    //from OpticalFlow
                            cv::Matx34d& P,
                            cv::Matx34d& P1,
                            std::vector<cv::DMatch>& matches,
                            std::vector<CloudPoint>& outCloud);
    
    /*****Triangulation.h*****/
    double TriangulatePoints(const vector<KeyPoint>& pt_set1, 
                            const vector<KeyPoint>& pt_set2,
                            const Mat& K,
                            const Mat& Kinv,
                            const Mat& distcoeff,
                            const Matx34d& P,
                            const Matx34d& P1,
                            vector<CloudPoint>& pointcloud,
                            vector<KeyPoint>& correspImg1Pt);
    /*****RichFeatureMatcher.cpp*****/
    void MatchFeatures(vector<cv::Mat>& imgs_,
                        vector<std::vector<cv::KeyPoint> >& imgpts_,
                        int idx_i,
                        int idx_j,
                        vector<DMatch>*matches=NULL);
                       
};

/*****Common.h*****/
void sfmSimpleApp::KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void sfmSimpleApp::PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
	kps.clear();
	for (unsigned int i=0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i],1.0f));
}

void sfmSimpleApp::GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2) 
{
	for (unsigned int i=0; i<matches.size(); i++) {
//		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
		assert(matches[i].queryIdx < imgpts1.size());
		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		assert(matches[i].trainIdx < imgpts2.size());
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
	}	
}

std::vector<cv::Point3d> sfmSimpleApp::CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

/*****RichFeatureMatcher.cpp*****/
void sfmSimpleApp::MatchFeatures(vector<cv::Mat>& imgs_,
                    vector<std::vector<cv::KeyPoint> >& imgpts_,
                    int idx_i,
                    int idx_j,
                    vector<DMatch>*matches)
{
    cv::Ptr<cv::FeatureDetector> detector = FeatureDetector::create("PyramidFAST");
	cv::Ptr<cv::DescriptorExtractor> extractor = DescriptorExtractor::create("ORB");
    //setup data
    std::vector<cv::Mat>& imgs = imgs_;
    std::vector<std::vector<cv::KeyPoint>>& imgpts = imgpts_;
    std::vector<cv::Mat> descriptors;
	
	std::cout << " -------------------- extract feature points for all images -------------------\n";
	detector->detect(imgs, imgpts);
	extractor->compute(imgs, imgpts, descriptors);
	std::cout << " ------------------------------------- done -----------------------------------\n";
    
    const vector<KeyPoint>& imgpts1 = imgpts[idx_i];
    const vector<KeyPoint>& imgpts2 = imgpts[idx_j];
    const Mat& descriptors_1 = descriptors[idx_i];
    const Mat& descriptors_2 = descriptors[idx_j];
    
    std::vector< DMatch > good_matches_,very_good_matches_;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    
    stringstream ss; ss << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << endl;
    cout << ss.str();
	stringstream ss1; ss1 << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << endl;
    cout << ss1.str();
    
    keypoints_1 = imgpts1;
    keypoints_2 = imgpts2;
    
    if(descriptors_1.empty())
    {
        CV_Error(0,"descriptors_1 is empty");
    }
    if(descriptors_2.empty())
    {
        CV_Error(0,"descriptors_2 is empty");
    }
    
    //matching descriptor vectors using Brute Force matcher
    BFMatcher matcher(NORM_HAMMING,true); //allow cross-check. use Hamming distance for binary descriptor (ORB)
    std::vector< DMatch > matches_;
    if (matches == NULL)
    {
        matches = &matches_;
    }
    if (matches->size() == 0)
    {
        matcher.match( descriptors_1, descriptors_2, *matches );
    }
    
	assert(matches->size() > 0);
    
    vector<KeyPoint> imgpts1_good,imgpts2_good;
    
    std::set<int> existing_trainIdx;
    for(unsigned int i = 0; i < matches->size(); i++ )
    { 
        //"normalize" matching: somtimes imgIdx is the one holding the trainIdx
        if ((*matches)[i].trainIdx <= 0) {
            (*matches)[i].trainIdx = (*matches)[i].imgIdx;
        }
        
        if( existing_trainIdx.find((*matches)[i].trainIdx) == existing_trainIdx.end() && 
           (*matches)[i].trainIdx >= 0 && (*matches)[i].trainIdx < (int)(keypoints_2.size()) /*&&
           (*matches)[i].distance > 0.0 && (*matches)[i].distance < cutoff*/ ) 
        {
            good_matches_.push_back( (*matches)[i]);
            imgpts1_good.push_back(keypoints_1[(*matches)[i].queryIdx]);
            imgpts2_good.push_back(keypoints_2[(*matches)[i].trainIdx]);
            existing_trainIdx.insert((*matches)[i].trainIdx);
        }
    }
    
    vector<uchar> status;
    vector<KeyPoint> imgpts2_very_good,imgpts1_very_good;
    
    assert(imgpts1_good.size() > 0);
	assert(imgpts2_good.size() > 0);
	assert(good_matches_.size() > 0);
	assert(imgpts1_good.size() == imgpts2_good.size() && imgpts1_good.size() == good_matches_.size());
	
    //Select features that make epipolar sense
    sfmSimpleApp::GetFundamentalMat(keypoints_1,keypoints_2,imgpts1_very_good,imgpts2_very_good,good_matches_);
}

/*****Triangulation.h*****/
double sfmSimpleApp::TriangulatePoints(const vector<KeyPoint>& pt_set1,
						const vector<KeyPoint>& pt_set2, 
						const Mat& K,
						const Mat& Kinv,
						const Mat& distcoeff,
						const Matx34d& P,
						const Matx34d& P1,
						vector<CloudPoint>& pointcloud,
						vector<KeyPoint>& correspImg1Pt)
{
	correspImg1Pt.clear();
	
	Matx44d P1_(P1(0,0),P1(0,1),P1(0,2),P1(0,3),
				P1(1,0),P1(1,1),P1(1,2),P1(1,3),
				P1(2,0),P1(2,1),P1(2,2),P1(2,3),
				0,		0,		0,		1);
	Matx44d P1inv(P1_.inv());
	
	cout << "Triangulating...";
	double t = getTickCount();
	vector<double> reproj_error;
	unsigned int pts_size = pt_set1.size();
    
    /***Triangulation:***/
    
    vector<Point2f> _pt_set1_pt,_pt_set2_pt;
	KeyPointsToPoints(pt_set1,_pt_set1_pt);
	KeyPointsToPoints(pt_set2,_pt_set2_pt);
	
	//undistort
	Mat pt_set1_pt,pt_set2_pt;
	undistortPoints(_pt_set1_pt, pt_set1_pt, K, distcoeff);
	undistortPoints(_pt_set2_pt, pt_set2_pt, K, distcoeff);
	
	//triangulate
	Mat pt_set1_pt_2r = pt_set1_pt.reshape(1, 2);
	Mat pt_set2_pt_2r = pt_set2_pt.reshape(1, 2);
	Mat pt_3d_h(1,pts_size,CV_32FC4);
	cv::triangulatePoints(P,P1,pt_set1_pt_2r,pt_set2_pt_2r,pt_3d_h);

	//calculate reprojection
	vector<Point3f> pt_3d;
	convertPointsHomogeneous(pt_3d_h.reshape(4, 1), pt_3d);
	cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P(0,0),P(0,1),P(0,2), P(1,0),P(1,1),P(1,2), P(2,0),P(2,1),P(2,2));
    cv::Vec3d rvec;
    Rodrigues(R ,rvec);
	cv::Vec3d tvec(P(0,3),P(1,3),P(2,3));
    
	vector<Point2f> reprojected_pt_set1;
	projectPoints(pt_3d,rvec,tvec,K,distcoeff,reprojected_pt_set1);

	for (unsigned int i=0; i<pts_size; i++) {
		CloudPoint cp; 
		cp.pt = pt_3d[i];
		pointcloud.push_back(cp);
		reproj_error.push_back(norm(_pt_set1_pt[i]-reprojected_pt_set1[i]));
	}
    
    Scalar mse = mean(reproj_error);
	cout << "Done. ("<<pointcloud.size()<<"points, " << t <<"s, mean reproj err = " << mse[0] << ")"<< endl;
    
    return mse[0];
}

bool sfmSimpleApp::CheckCoherentRotation(cv::Mat_<double>& R) {
	if(fabsf(determinant(R))-1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}
	return true;
}

/******FindCameraMatrices.cpp*****/
void sfmSimpleApp::TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
    //Using OpenCV's SVD
	SVD svd(E,SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
    cout << "----------------------- SVD ------------------------\n";
	cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
	cout << "----------------------------------------------------\n";
}

bool sfmSimpleApp::DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2) 
{
    //Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		cout << "singular values are too far apart\n";
		return false;
	}

	Matx33d W(0,-1,0,	//HZ 9.13
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3

    return true;
}

Mat sfmSimpleApp::GetFundamentalMat(const vector<KeyPoint>& imgpts1,
					   const vector<KeyPoint>& imgpts2,
					   vector<KeyPoint>& imgpts1_good,
					   vector<KeyPoint>& imgpts2_good,
					   vector<DMatch>& matches)
{
    //Try to eliminate keypoints based on the fundamental matrix
	//(although this is not the proper way to do this)
	vector<uchar>status(imgpts1.size());
    cout << "imgpts1.size()" <<imgpts1.size();
    imgpts1_good.clear(); imgpts2_good.clear();
	
	vector<KeyPoint> imgpts1_tmp;
	vector<KeyPoint> imgpts2_tmp;
	if (matches.size() <= 0)
    {
		//points already aligned...
		imgpts1_tmp = imgpts1;
		imgpts2_tmp = imgpts2;
        cout << "points already aligned" <<endl;
	}
    else
    {
		GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
	}
    
    Mat F;
    {
		vector<Point2f> pts1,pts2;
		KeyPointsToPoints(imgpts1_tmp, pts1);
		KeyPointsToPoints(imgpts2_tmp, pts2);
        double minVal,maxVal;
		cv::minMaxIdx(pts1,&minVal,&maxVal);
		F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
	}
    
    vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;	
	for (unsigned int i=0; i<status.size(); i++)
    {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			if (matches.size() <= 0){ //points already aligned...
				new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
			} else {
				new_matches.push_back(matches[i]);
			}
        }
    }
    
    cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
	matches = new_matches; //keep only those points who survived the fundamental matrix
    return F;
}


bool sfmSimpleApp::TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status)
{
	vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	Matx44d P4x4 = Matx44d::eye(); 
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];
	
	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	if(false) //not
	{
		cv::Mat_<double> cldm(pcloud.size(),3);
		for(unsigned int i=0;i<pcloud.size();i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for (int i=0; i<pcloud.size(); i++) {
			cv::Vec3d w = cv::Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if(D < p_to_plane_thresh) num_inliers++;
		}

		cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
		if((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return false;
	}

	return true;
}




bool sfmSimpleApp::FindCameraMatrices(const Mat& K,
						const Mat& Kinv, 
						const Mat& distcoeff,
						const vector<KeyPoint>& imgpts1, //imgpts1
						const vector<KeyPoint>& imgpts2, //imgpts2
						vector<KeyPoint>& imgpts1_good,  //imgpts_good1
						vector<KeyPoint>& imgpts2_good,  //imgpts_good2
						Matx34d& P,                      //input identity
						Matx34d& P1,                     //input identity, output position
						vector<DMatch>& matches,         //from RichFeaturesMatcher, returned modified
						vector<CloudPoint>& outCloud)
{
    Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches);
    if(matches.size() < 100)
    {
        cerr << "not enough inliers after F matrix" << endl;
        return false;
    }
    
    //Essential matrix: compute then extract cameras [R|t]
    Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

    //according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
    if(fabsf(determinant(E)) > 1e-07) {
        cout << "det(E) != 0 : " << determinant(E) << "\n";
        P1 = 0;
        return false;
    }
    
    Mat_<double> R1(3,3);
    Mat_<double> R2(3,3);
    Mat_<double> t1(1,3);
    Mat_<double> t2(1,3);
    
    //decompose E to P' , HZ (9.19)
    if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

    //check validity of results
    if(determinant(R1)+1.0 < 1e-09)
    {
        //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
        cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
        E = -E;
        DecomposeEtoRandT(E,R1,R2,t1,t2);
    }
    if (!CheckCoherentRotation(R1))
    {
        cout << "resulting rotation is not coherent\n";
        P1 = 0;
        return false;
    }

    //test 4 configurations of P'
    
    //CONFIGURATION 1
    P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
                R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
                R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
    cout << "Testing P1 " << endl << Mat(P1) << endl;
    
    vector<CloudPoint> pcloud,pcloud1; vector<KeyPoint> corresp;
    double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
    double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
    vector<uchar> tmp_status;
    //check if pointa are triangulated --in front-- of cameras for all 4 ambiguations

    //CONFIGURATION 2
    if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
        P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
                     R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
                     R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
        cout << "Testing P1 "<< endl << Mat(P1) << endl;

        pcloud.clear(); pcloud1.clear(); corresp.clear();
        reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
        reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
        
        //CONFIGURATION 3
        if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
            if (!CheckCoherentRotation(R2)) {
                cout << "resulting rotation is not coherent\n";
                P1 = 0;
                return false;
            }
            
            P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
                         R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
                         R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
            cout << "Testing P1 "<< endl << Mat(P1) << endl;

            pcloud.clear(); pcloud1.clear(); corresp.clear();
            reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
            reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
            
            //CONFIGURATION 4
            if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
                P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
                             R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
                             R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
                cout << "Testing P1 "<< endl << Mat(P1) << endl;

                pcloud.clear(); pcloud1.clear(); corresp.clear();
                reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
                reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
                
                if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
                    cout << "Shit." << endl; 
                    return false;
                }
            }
        }
    }
    
    for (unsigned int i=0; i<pcloud.size(); i++)
    {
        outCloud.push_back(pcloud[i]);
    }
    
    return true;
}

void sfmSimpleApp::GetPosition(const Mat& frame1, const Mat& frame2)
{
    if (!features_matched)
    {
        imgpts1.clear(); imgpts2.clear(); fullpts1.clear(); fullpts2.clear();
        std::vector<cv::Mat> imgs;
        imgs.push_back(frame1); imgs.push_back(frame2);
    
        std::vector<std::vector<cv::KeyPoint>> imgpts;
        imgpts.push_back(imgpts1); imgpts.push_back(imgpts2);
        
        MatchFeatures(imgs, imgpts, 0, 1);
        imgpts1 = imgpts.at(0);
        imgpts2 = imgpts.at(1);
        features_matched = true;
    }
    
    std::vector<cv::DMatch> matches;
    FindCameraMatrices(K, Kinv, distortion_coeff, imgpts1, imgpts2, imgpts1_good, imgpts2_good, P, P1, matches, pcloud);
    cout << "P0: " << P << "P1: " << P1 << endl;
}

void sfmSimpleApp::setup()
{
    features_matched = false;
    P = cv::Matx34d(1,0,0,0,
                    0,1,0,0,
                    0,0,1,0);
    P1 = cv::Matx34d(   1,0,0,50,
                        0,1,0,0,
                        0,0,1,0);
    
    cv::FileStorage fs;
    fs.open("/users/wolfgag/indoorPositioning/sfmSimple/out_camera_data.xml",cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Failed to open";
        return;
    }
    
    fs["Camera_Matrix"]>>cam_matrix;
    fs["Distortion_Coefficients"]>>distortion_coeff;
    
    cout << "cam_matrix:" << cam_matrix;
    K = cam_matrix;
    invert(K, Kinv);
    
    capture.open("/Users/wolfgag/indoorPositioning/ios_test.MOV");

}

void sfmSimpleApp::update()
{
    if (!capture.isOpened()) return;
    if (capture.grab())
    {
        cv::Mat i_mFrame1;
        cv::Mat mFrame1;
    
        capture >> i_mFrame1;
        i_mFrame1.copyTo(mFrame1);
		mTexture = gl::Texture(fromOcv(i_mFrame1));
    
        cvtColor(i_mFrame1, mFrame1, CV_BGR2GRAY);
		if( mFrame0.data && getElapsedFrames() % 30 == 0)
        {
            cout << "let's get position!" << endl;
            GetPosition(mFrame0, mFrame1);
		}
        
		mFrame1.copyTo(mFrame0);
	}
}

void sfmSimpleApp::draw()
{
    if (!capture.isOpened() || !mTexture.getWidth()) return;
	gl::clear();
    gl::setMatricesWindow( getWindowSize() );
	gl::draw( mTexture );
}

CINDER_APP_NATIVE( sfmSimpleApp, RendererGl )
