#include <atomic>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include "System.h"

static std::atomic<bool> running(true);
static void on_signal(int) { running = false; }

static std::string num(double v, int p=10) {
    std::ostringstream s; s << std::fixed << std::setprecision(p) << v; return s.str();
}

static bool key_match(const std::string& line, const std::string& key) {
    size_t i=line.find_first_not_of(" \t");
    if(i==std::string::npos || line.compare(i,key.size(),key)!=0) return false;
    i+=key.size(); while(i<line.size() && (line[i]==' '||line[i]=='\t')) ++i;
    return i<line.size() && line[i]==':';
}

static float baseline_m(const rs2::device& dev) {
    for(auto s: dev.query_sensors()) {
        try {
            if(s.supports(RS2_OPTION_STEREO_BASELINE))
                return s.get_option(RS2_OPTION_STEREO_BASELINE)*0.001f;
        } catch(...) {}
    }
    return -1.f;
}

static const char* distortion_name(rs2_distortion m) {
    switch(m) {
        case RS2_DISTORTION_NONE: return "None";
        case RS2_DISTORTION_MODIFIED_BROWN_CONRADY: return "Modified Brown-Conrady";
        case RS2_DISTORTION_INVERSE_BROWN_CONRADY: return "Inverse Brown-Conrady";
        case RS2_DISTORTION_BROWN_CONRADY: return "Brown-Conrady";
        case RS2_DISTORTION_FTHETA: return "F-Theta";
        case RS2_DISTORTION_KANNALA_BRANDT4: return "Kannala-Brandt4";
        default: return "Unknown";
    }
}

static void write_runtime_yaml(const std::string& src, const std::string& dst,
                               const rs2_intrinsics& k, float depth_scale,
                               float b, int fps) {
    std::ifstream in(src);
    if(!in) throw std::runtime_error("Cannot open YAML template: "+src);

    std::map<std::string,std::string> v;
    v["Camera1.fx"]=num(k.fx); v["Camera1.fy"]=num(k.fy);
    v["Camera1.cx"]=num(k.ppx); v["Camera1.cy"]=num(k.ppy);
    v["Camera1.k1"]="0.0"; v["Camera1.k2"]="0.0";
    v["Camera1.p1"]="0.0"; v["Camera1.p2"]="0.0";
    v["Camera.width"]=std::to_string(k.width);
    v["Camera.height"]=std::to_string(k.height);
    v["Camera.fps"]=std::to_string(fps);
    v["Camera.RGB"]="1";
    v["RGBD.DepthMapFactor"]=num(1.0/depth_scale,6);
    if(b>0) v["Stereo.b"]=num(b,8);

    std::ofstream out(dst);
    if(!out) throw std::runtime_error("Cannot write runtime YAML: "+dst);

    std::string line;
    while(std::getline(in,line)) {
        bool done=false;
        for(const auto& kv:v) if(key_match(line,kv.first)) {
            out << kv.first << ": " << kv.second << "\n"; done=true; break;
        }
        if(!done) out << line << "\n";
    }
}

static void make_maps(const rs2_intrinsics& k, cv::Mat& mx, cv::Mat& my, bool& use) {
    bool zero=true; for(int i=0;i<5;i++) zero &= std::fabs(k.coeffs[i])<1e-10f;
    use = (k.model!=RS2_DISTORTION_NONE && !zero);
    if(!use) return;

    mx.create(k.height,k.width,CV_32FC1);
    my.create(k.height,k.width,CV_32FC1);

    for(int y=0;y<k.height;y++) {
        float* px=mx.ptr<float>(y); float* py=my.ptr<float>(y);
        for(int x=0;x<k.width;x++) {
            float p[3]={(x-k.ppx)/k.fx,(y-k.ppy)/k.fy,1.f};
            float q[2]; rs2_project_point_to_pixel(q,&k,p);
            px[x]=q[0]; py[x]=q[1];
        }
    }
}

int main(int argc,char** argv) {
    if(argc<3 || argc>4) {
        std::cerr<<"Usage: "<<argv[0]<<" vocabulary template_yaml [trajectory_prefix]\n";
        return 1;
    }
    std::signal(SIGINT,on_signal); std::signal(SIGTERM,on_signal);

    try {
        rs2::context ctx; auto devs=ctx.query_devices();
        if(devs.size()==0) throw std::runtime_error("No RealSense device connected");

        rs2::device dev=devs[0];
        std::string name=dev.supports(RS2_CAMERA_INFO_NAME)?dev.get_info(RS2_CAMERA_INFO_NAME):"RealSense";
        std::string serial=dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)?dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER):"unknown";
        std::cout<<"Device: "<<name<<"\nSerial: "<<serial<<"\n";

        const int W=640,H=480,FPS=30;
        rs2::pipeline pipe; rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR,W,H,RS2_FORMAT_RGB8,FPS);
        cfg.enable_stream(RS2_STREAM_DEPTH,W,H,RS2_FORMAT_Z16,FPS);
        auto profile=pipe.start(cfg);

        auto cp=profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        rs2_intrinsics k=cp.get_intrinsics();
        auto ds=profile.get_device().first<rs2::depth_sensor>();
        float depth_scale=ds.get_depth_scale();
        float b=baseline_m(profile.get_device());

        std::cout<<"\nActive calibration\n"
                 <<" fx/fy: "<<k.fx<<" / "<<k.fy<<"\n"
                 <<" cx/cy: "<<k.ppx<<" / "<<k.ppy<<"\n"
                 <<" size : "<<k.width<<" x "<<k.height<<"\n"
                 <<" model: "<<distortion_name(k.model)<<"\n"
                 <<" coeff: "<<k.coeffs[0]<<", "<<k.coeffs[1]<<", "<<k.coeffs[2]<<", "<<k.coeffs[3]<<", "<<k.coeffs[4]<<"\n"
                 <<" depth scale: "<<depth_scale<<" m/unit\n"
                 <<" baseline: "<<b<<" m\n";

        cv::Mat mx,my; bool remap=false;
        std::cout<<"Building rectification map..."<<std::flush;
        make_maps(k,mx,my,remap);
        std::cout<<(remap?" enabled\n":" not needed\n");

        const std::string runtime_yaml="Examples/RGB-D/RealSense_D400_runtime.yaml";
        write_runtime_yaml(argv[2],runtime_yaml,k,depth_scale,b,FPS);
        std::cout<<"Runtime YAML: "<<runtime_yaml<<"\n";

        rs2::align align(RS2_STREAM_COLOR);
        ORB_SLAM3::System SLAM(argv[1],runtime_yaml,ORB_SLAM3::System::RGBD,true);

        while(running && !SLAM.isShutDown()) {
            auto fs=align.process(pipe.wait_for_frames());
            auto color=fs.get_color_frame();
            auto depth=fs.get_depth_frame();
            if(!color || !depth) continue;

            cv::Mat c0(cv::Size(k.width,k.height),CV_8UC3,
                       const_cast<void*>(color.get_data()),cv::Mat::AUTO_STEP);
            cv::Mat d0(cv::Size(k.width,k.height),CV_16UC1,
                       const_cast<void*>(depth.get_data()),cv::Mat::AUTO_STEP);
            cv::Mat c,d;
            if(remap) {
                cv::remap(c0,c,mx,my,cv::INTER_LINEAR,cv::BORDER_CONSTANT);
                cv::remap(d0,d,mx,my,cv::INTER_NEAREST,cv::BORDER_CONSTANT);
            } else { c=c0; d=d0; }

            SLAM.TrackRGBD(c,d,color.get_timestamp()*1e-3);
        }

        pipe.stop();
        SLAM.Shutdown();

        std::string prefix=argc==4?argv[3]:"RealSense";
        SLAM.SaveTrajectoryTUM(prefix+"_CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM(prefix+"_KeyFrameTrajectory.txt");
        return 0;
    }
    catch(const rs2::error& e) {
        std::cerr<<"RealSense error: "<<e.what()<<"\n"; return 2;
    }
    catch(const std::exception& e) {
        std::cerr<<"Error: "<<e.what()<<"\n"; return 3;
    }
}
