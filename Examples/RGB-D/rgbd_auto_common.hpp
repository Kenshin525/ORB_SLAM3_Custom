#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Config.h"
#include "System.h"

namespace rgbd_auto {

using Clock = std::chrono::steady_clock;
static std::atomic<bool> g_running(true);
static inline void signal_handler(int) { g_running = false; }

struct CameraCalibration {
    std::string backend, name, serial, alignment_note, calibration_note;
    std::string intrinsics_source, distortion_source, depth_scale_source, alignment_source;
    int width=0, height=0, fps=30;
    double fx=0, fy=0, cx=0, cy=0;
    std::array<double,5> dist{{0,0,0,0,0}}; // k1,k2,p1,p2,k3
    double depth_scale_m=0.001;             // meters / raw unit
    bool strict_ok=false;
};

struct RGBDFrame {
    cv::Mat rgb;    // CV_8UC3, RGB order
    cv::Mat depth;  // CV_16UC1, raw depth units
    double timestamp_s=0;
    uint64_t frame_number=0;
};

class CameraSource {
public:
    virtual ~CameraSource() {}
    virtual void start()=0;
    virtual bool grab(RGBDFrame& out)=0;
    virtual void stop()=0;
    virtual CameraCalibration calibration() const=0;
    virtual std::string metadata() const { return std::string(); }
};

static inline std::string num(double v,int p=10) {
    std::ostringstream s; s<<std::fixed<<std::setprecision(p)<<v; return s.str();
}
static inline bool key_match(const std::string& line,const std::string& key) {
    size_t i=line.find_first_not_of(" \t");
    if(i==std::string::npos || line.compare(i,key.size(),key)!=0) return false;
    i+=key.size(); while(i<line.size()&&(line[i]==' '||line[i]=='\t')) ++i;
    return i<line.size() && line[i]==':';
}
static inline bool read_yaml_scalar(const std::string& path,const std::string& key,double& value) {
    std::ifstream in(path.c_str()); if(!in) return false; std::string line;
    while(std::getline(in,line)) if(key_match(line,key)) {
        auto c=line.find(':'); if(c==std::string::npos) return false;
        try { value=std::stod(line.substr(c+1)); return true; } catch(...) { return false; }
    }
    return false;
}
static inline std::string sanitize_name(std::string s) {
    for(char& c:s) if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='-'||c=='_'||c=='.')) c='_';
    if(s.empty()) s="RGBD"; return s;
}
static inline void make_dir_if_needed(const std::string& p) {
    if(::mkdir(p.c_str(),0755)==0 || errno==EEXIST) return;
    throw std::runtime_error("Cannot create directory '"+p+"': "+std::strerror(errno));
}
static inline std::string create_session_dir(const std::string& prefix) {
    make_dir_if_needed("trajectories"); std::time_t now=std::time(NULL); std::tm t; localtime_r(&now,&t);
    char stamp[32]; std::strftime(stamp,sizeof(stamp),"%Y%m%d_%H%M%S",&t);
    std::string base="trajectories/"+sanitize_name(prefix)+"_"+stamp;
    for(int i=0;i<1000;++i) { std::string p=i?base+"_"+std::to_string(i):base; if(::mkdir(p.c_str(),0755)==0) return p; if(errno!=EEXIST) break; }
    throw std::runtime_error("Could not create unique session directory");
}


static inline bool finite_number(double v) { return std::isfinite(v); }
static inline void validate_strict_calibration(const CameraCalibration& c) {
    std::vector<std::string> bad;
    if(!c.strict_ok) bad.push_back("backend did not certify factory/runtime calibration");
    if(c.width<=0 || c.height<=0 || c.fps<=0) bad.push_back("active resolution/fps");
    if(!finite_number(c.fx) || !finite_number(c.fy) || c.fx<=0 || c.fy<=0) bad.push_back("real fx/fy");
    if(!finite_number(c.cx) || !finite_number(c.cy) || c.cx<0 || c.cy<0 || c.cx>=c.width || c.cy>=c.height) bad.push_back("real cx/cy");
    if(!finite_number(c.depth_scale_m) || c.depth_scale_m<=0) bad.push_back("real depth scale");
    if(c.intrinsics_source.empty()) bad.push_back("intrinsics source");
    if(c.distortion_source.empty()) bad.push_back("distortion source");
    if(c.depth_scale_source.empty()) bad.push_back("depth scale source");
    if(c.alignment_source.empty()) bad.push_back("alignment source");
    if(!bad.empty()) {
        std::ostringstream ss;
        ss << "STRICT CALIBRATION: FAIL\n";
        for(const auto& x: bad) ss << "  invalid/missing: " << x << "\n";
        ss << "Approximate/fallback sensor calibration is disabled.";
        throw std::runtime_error(ss.str());
    }
}
static inline void print_strict_calibration(const CameraCalibration& c) {
    std::cout << "\n=== STRICT CALIBRATION AUDIT ===\n"
              << "Camera                  : " << c.name << "\n"
              << "Backend                 : " << c.backend << "\n"
              << "Serial                  : " << c.serial << "\n"
              << "Intrinsics source       : " << c.intrinsics_source << "\n"
              << "Distortion source       : " << c.distortion_source << "\n"
              << "Depth scale source      : " << c.depth_scale_source << "\n"
              << "Alignment source        : " << c.alignment_source << "\n"
              << "Fallback calibration    : DISABLED\n"
              << "STRICT CALIBRATION      : PASS\n"
              << "================================\n\n";
}

static inline void write_runtime_yaml(const std::string& src,const std::string& dst,const CameraCalibration& c) {
    if(c.depth_scale_m<=0 || c.fx<=0 || c.fy<=0 || c.width<=0 || c.height<=0) throw std::runtime_error("Invalid runtime calibration");
    std::ifstream in(src.c_str()); if(!in) throw std::runtime_error("Cannot open YAML template: "+src);
    std::map<std::string,std::string> v;
    v["Camera1.fx"]=num(c.fx); v["Camera1.fy"]=num(c.fy); v["Camera1.cx"]=num(c.cx); v["Camera1.cy"]=num(c.cy);
    v["Camera1.k1"]="0.0"; v["Camera1.k2"]="0.0"; v["Camera1.p1"]="0.0"; v["Camera1.p2"]="0.0";
    v["Camera.width"]=std::to_string(c.width); v["Camera.height"]=std::to_string(c.height); v["Camera.fps"]=std::to_string(c.fps);
    v["Camera.RGB"]="1"; v["RGBD.DepthMapFactor"]=num(1.0/c.depth_scale_m,6);
    std::ofstream out(dst.c_str()); if(!out) throw std::runtime_error("Cannot write runtime YAML: "+dst);
    std::string line; while(std::getline(in,line)) { bool replaced=false; for(const auto& kv:v) if(key_match(line,kv.first)) { out<<kv.first<<": "<<kv.second<<"\n"; replaced=true; break; } if(!replaced) out<<line<<"\n"; }
}

static inline void make_undistort_maps(const CameraCalibration& c,cv::Mat& mx,cv::Mat& my,bool& use) {
    use=false; for(double d:c.dist) if(std::fabs(d)>1e-10) use=true; if(!use) return;
    cv::Mat K=(cv::Mat_<double>(3,3)<<c.fx,0,c.cx,0,c.fy,c.cy,0,0,1);
    cv::Mat D=(cv::Mat_<double>(1,5)<<c.dist[0],c.dist[1],c.dist[2],c.dist[3],c.dist[4]);
    cv::initUndistortRectifyMap(K,D,cv::Mat(),K,cv::Size(c.width,c.height),CV_32FC1,mx,my);
}

static inline const char* tracking_state_name(int s) {
    switch(s) {
        case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:return "SYSTEM_NOT_READY";
        case ORB_SLAM3::Tracking::NO_IMAGES_YET:return "NO_IMAGES_YET";
        case ORB_SLAM3::Tracking::NOT_INITIALIZED:return "NOT_INITIALIZED";
        case ORB_SLAM3::Tracking::OK:return "OK";
        case ORB_SLAM3::Tracking::RECENTLY_LOST:return "RECENTLY_LOST";
        case ORB_SLAM3::Tracking::LOST:return "LOST";
        case ORB_SLAM3::Tracking::OK_KLT:return "OK_KLT";
        default:return "UNKNOWN";
    }
}
struct PoseData { double x=0,y=0,z=0,qx=0,qy=0,qz=0,qw=1,roll_deg=0,pitch_deg=0,yaw_deg=0; };
static inline PoseData pose_from_Tcw(const Sophus::SE3f& Tcw) {
    Sophus::SE3f Twc=Tcw.inverse(); Eigen::Vector3f t=Twc.translation(); Eigen::Quaternionf q=Twc.unit_quaternion(); q.normalize(); Eigen::Matrix3f R=q.toRotationMatrix();
    const double k=180.0/3.14159265358979323846; PoseData p; p.x=t.x();p.y=t.y();p.z=t.z(); p.qx=q.x();p.qy=q.y();p.qz=q.z();p.qw=q.w();
    p.roll_deg=std::atan2((double)R(2,1),(double)R(2,2))*k;
    p.pitch_deg=std::atan2(-(double)R(2,0),std::sqrt((double)R(0,0)*R(0,0)+(double)R(1,0)*R(1,0)))*k;
    p.yaw_deg=std::atan2((double)R(1,0),(double)R(0,0))*k; return p;
}
static inline void write_live_row(std::ofstream& f,uint64_t n,double ts,int state,bool valid,const PoseData& p) {
    f<<n<<","<<std::fixed<<std::setprecision(9)<<ts<<","<<tracking_state_name(state)<<","<<(valid?1:0)<<",";
    if(valid) f<<p.x<<","<<p.y<<","<<p.z<<","<<p.qx<<","<<p.qy<<","<<p.qz<<","<<p.qw<<","<<p.roll_deg<<","<<p.pitch_deg<<","<<p.yaw_deg;
    else f<<"nan,nan,nan,nan,nan,nan,nan,nan,nan,nan"; f<<"\n"; f.flush();
}
static inline void quaternion_to_rpy_deg(double qx,double qy,double qz,double qw,double& r,double& p,double& y) {
    Eigen::Quaterniond q(qw,qx,qy,qz);q.normalize();Eigen::Matrix3d R=q.toRotationMatrix(); const double k=180.0/3.14159265358979323846;
    r=std::atan2(R(2,1),R(2,2))*k; p=std::atan2(-R(2,0),std::sqrt(R(0,0)*R(0,0)+R(1,0)*R(1,0)))*k; y=std::atan2(R(1,0),R(0,0))*k;
}
static inline bool tum_to_csv(const std::string& tum,const std::string& csv) {
    std::ifstream in(tum.c_str()); std::ofstream out(csv.c_str()); if(!in||!out) return false;
    out<<"timestamp_s,x_m,y_m,z_m,qx,qy,qz,qw,roll_deg,pitch_deg,yaw_deg\n"; double ts,x,y,z,qx,qy,qz,qw;
    while(in>>ts>>x>>y>>z>>qx>>qy>>qz>>qw) { double r,p,yy; quaternion_to_rpy_deg(qx,qy,qz,qw,r,p,yy); out<<std::fixed<<std::setprecision(9)<<ts<<","<<x<<","<<y<<","<<z<<","<<qx<<","<<qy<<","<<qz<<","<<qw<<","<<r<<","<<p<<","<<yy<<"\n"; }
    return true;
}
struct SegmentMeta { int id=0; std::string dir,live_csv_path,final_csv_path,final_tum_path,keyframe_tum_path,info_path; double start_ts=0,end_ts=0; uint64_t frames=0,valid_frames=0; };
static inline std::string make_segment_dir(const std::string& session,int id) { std::ostringstream s;s<<session<<"/segment_"<<std::setw(3)<<std::setfill('0')<<id;make_dir_if_needed(s.str());return s.str(); }
static inline void write_segment_info(const SegmentMeta& s) { std::ofstream f(s.info_path.c_str()); if(!f)return; f<<std::fixed<<std::setprecision(9)<<"segment_id="<<s.id<<"\nstart_timestamp_s="<<s.start_ts<<"\nend_timestamp_s="<<s.end_ts<<"\nframes_recorded="<<s.frames<<"\nvalid_pose_frames="<<s.valid_frames<<"\nlive_csv="<<s.live_csv_path<<"\nfinal_csv="<<s.final_csv_path<<"\nfinal_tum="<<s.final_tum_path<<"\nkeyframe_tum="<<s.keyframe_tum_path<<"\n"; }
static inline bool filter_tum_range(const std::string& inpath,const std::string& outpath,double a,double b) { std::ifstream in(inpath.c_str());std::ofstream out(outpath.c_str());if(!in||!out)return false;std::string line;bool any=false;while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::istringstream ss(line);double t;if(!(ss>>t))continue;if(t+1e-6<a||t-1e-6>b)continue;out<<line<<"\n";any=true;}return any; }
static inline void write_session_info(const std::string& path,const CameraCalibration& c,const std::string& template_yaml,const std::string& runtime_yaml,const std::string& extra) {
    std::ofstream f(path.c_str()); if(!f)return; f<<std::setprecision(12);
    f<<"strict_calibration=PASS\nfallback_sensor_calibration=DISABLED\nbackend="<<c.backend<<"\ndevice_name="<<c.name<<"\nserial="<<c.serial<<"\nwidth="<<c.width<<"\nheight="<<c.height<<"\nfps="<<c.fps<<"\nfx="<<c.fx<<"\nfy="<<c.fy<<"\ncx="<<c.cx<<"\ncy="<<c.cy<<"\ndistortion_coeffs="<<c.dist[0]<<","<<c.dist[1]<<","<<c.dist[2]<<","<<c.dist[3]<<","<<c.dist[4]<<"\ndepth_scale_m_per_unit="<<c.depth_scale_m<<"\nalignment="<<c.alignment_note<<"\ncalibration_note="<<c.calibration_note<<"\ntemplate_yaml="<<template_yaml<<"\nruntime_yaml="<<runtime_yaml<<"\npose_convention=Twc_camera_pose_in_ORB_SLAM3_world\nrpy_convention=ZYX_yaw_pitch_roll_degrees_from_Rwc\nlive_csv_note=flushed_each_frame_and_may_differ_from_final_after_BA_or_loop_closure\nfinal_csv_note=derived_from_ORB_SLAM3_SaveTrajectoryTUM_after_shutdown_lost_frames_omitted\n";
    if(!extra.empty()) f<<extra<<"\n";
}

static inline int run(CameraSource& src,const std::string& vocabulary,const std::string& template_yaml,const std::string& prefix) {
    std::signal(SIGINT,signal_handler); std::signal(SIGTERM,signal_handler); g_running=true;
    src.start(); CameraCalibration c=src.calibration(); validate_strict_calibration(c); print_strict_calibration(c);
    const std::string session=create_session_dir(prefix.empty()?c.name:prefix); const std::string yaml=session+"/runtime.yaml"; const std::string info=session+"/session_info.txt";
    const std::string full=session+"/_internal_full_trajectory_tum.txt", fullkf=session+"/_internal_full_keyframe_tum.txt";
    write_runtime_yaml(template_yaml,yaml,c); write_session_info(info,c,template_yaml,yaml,src.metadata());
    double template_b=0; bool have_b=read_yaml_scalar(template_yaml,"Stereo.b",template_b);
    std::cout<<"Device: "<<c.name<<"\nBackend: "<<c.backend<<"\nSerial: "<<c.serial<<"\nSession directory: "<<session<<"\n"
             <<"Calibration: fx="<<c.fx<<" fy="<<c.fy<<" cx="<<c.cx<<" cy="<<c.cy<<" size="<<c.width<<"x"<<c.height<<" depth_scale="<<c.depth_scale_m<<" m/unit\n";
    if(have_b) std::cout<<"ORB-SLAM3 RGB-D Stereo.b: "<<template_b<<" m (kept from template)\n";

    cv::Mat mx,my; bool remap=false; make_undistort_maps(c,mx,my,remap);
    ORB_SLAM3::System SLAM(vocabulary,yaml,ORB_SLAM3::System::RGBD,true);
    std::cout<<"\nSLAM running. Viewer buttons: Start Save Trajectory / Stop Save Trajectory / Stop\n";

    bool recording=false; int next_id=1; std::ofstream csv; SegmentMeta cur; std::vector<SegmentMeta> done; uint64_t processed=0;
    auto start_seg=[&](double ts){ if(recording)return; cur=SegmentMeta();cur.id=next_id++;cur.dir=make_segment_dir(session,cur.id);cur.live_csv_path=cur.dir+"/camera_trajectory_live.csv";cur.final_csv_path=cur.dir+"/camera_trajectory_final.csv";cur.final_tum_path=cur.dir+"/camera_trajectory_final_tum.txt";cur.keyframe_tum_path=cur.dir+"/keyframe_trajectory_final_tum.txt";cur.info_path=cur.dir+"/segment_info.txt";cur.start_ts=cur.end_ts=ts;csv.open(cur.live_csv_path.c_str());if(!csv)throw std::runtime_error("Cannot open segment CSV");csv<<"frame_number,timestamp_s,tracking_state,valid,x_m,y_m,z_m,qx,qy,qz,qw,roll_deg,pitch_deg,yaw_deg\n";csv.flush();recording=true;SLAM.SetTrajectoryRecordingState(true);std::cout<<"\n[REC START] segment "<<cur.id<<" -> "<<cur.live_csv_path<<"\n";};
    auto stop_seg=[&](){ if(!recording){SLAM.SetTrajectoryRecordingState(false);return;}csv.flush();csv.close();write_segment_info(cur);done.push_back(cur);std::cout<<"\n[REC STOP] segment "<<cur.id<<" frames="<<cur.frames<<" valid="<<cur.valid_frames<<"\n";recording=false;SLAM.SetTrajectoryRecordingState(false);};

    while(g_running && !SLAM.isShutDown()) {
        RGBDFrame f; if(!src.grab(f)) continue; if(f.rgb.empty()||f.depth.empty()) continue;
        if(f.rgb.cols!=c.width||f.rgb.rows!=c.height) cv::resize(f.rgb,f.rgb,cv::Size(c.width,c.height),0,0,cv::INTER_LINEAR);
        if(f.depth.cols!=c.width||f.depth.rows!=c.height) cv::resize(f.depth,f.depth,cv::Size(c.width,c.height),0,0,cv::INTER_NEAREST);
        cv::Mat rgb=f.rgb,depth=f.depth,rr,dd; if(remap){cv::remap(rgb,rr,mx,my,cv::INTER_LINEAR,cv::BORDER_CONSTANT);cv::remap(depth,dd,mx,my,cv::INTER_NEAREST,cv::BORDER_CONSTANT);rgb=rr;depth=dd;}
        Sophus::SE3f Tcw=SLAM.TrackRGBD(rgb,depth,f.timestamp_s); int state=SLAM.GetTrackingState(); bool valid=state==ORB_SLAM3::Tracking::OK||state==ORB_SLAM3::Tracking::OK_KLT; PoseData pose;if(valid)pose=pose_from_Tcw(Tcw);
        int cmd=SLAM.ConsumeTrajectoryRecordCommand(); bool app_stop=false;
        if(cmd==ORB_SLAM3::System::TRAJECTORY_RECORD_START) start_seg(f.timestamp_s);
        else if(cmd==ORB_SLAM3::System::TRAJECTORY_RECORD_STOP) stop_seg();
        else if(cmd==ORB_SLAM3::System::TRAJECTORY_APP_STOP) app_stop=true;
        if(recording){write_live_row(csv,f.frame_number,f.timestamp_s,state,valid,pose);cur.end_ts=f.timestamp_s;++cur.frames;if(valid)++cur.valid_frames;}
        ++processed; if(processed%6==0){std::cout<<"\r[POSE] "<<(recording?"[REC] ":"[IDLE]")<<tracking_state_name(state);if(valid)std::cout<<std::fixed<<std::setprecision(3)<<" xyz[m]=("<<pose.x<<", "<<pose.y<<", "<<pose.z<<") rpy[deg]=("<<pose.roll_deg<<", "<<pose.pitch_deg<<", "<<pose.yaw_deg<<")";else std::cout<<" pose=INVALID";std::cout<<"          "<<std::flush;}
        if(app_stop){g_running=false;break;}
    }
    std::cout<<"\n"; if(recording) stop_seg(); src.stop();
    if(!SLAM.isShutDown()) SLAM.Shutdown();
    if(!done.empty()) { SLAM.SaveTrajectoryTUM(full); SLAM.SaveKeyFrameTrajectoryTUM(fullkf); for(auto& s:done){bool ok=filter_tum_range(full,s.final_tum_path,s.start_ts,s.end_ts);filter_tum_range(fullkf,s.keyframe_tum_path,s.start_ts,s.end_ts);if(ok)tum_to_csv(s.final_tum_path,s.final_csv_path);write_segment_info(s);} }
    std::cout<<"\nSESSION COMPLETE: "<<session<<"\nRecorded segments: "<<done.size()<<"\nUse camera_trajectory_final.csv as primary result.\n";
    return 0;
}

} // namespace rgbd_auto
