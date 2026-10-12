#include <algorithm>
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
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include "Config.h"
#include "System.h"

using Clock = std::chrono::steady_clock;
static std::atomic<bool> running(true);
static void on_signal(int) { running = false; }

static double ms_between(const Clock::time_point& a, const Clock::time_point& b)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b-a).count();
}

static std::string num(double v, int p=10)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(p) << v;
    return s.str();
}

static bool key_match(const std::string& line, const std::string& key)
{
    size_t i = line.find_first_not_of(" \t");
    if(i == std::string::npos || line.compare(i,key.size(),key) != 0) return false;
    i += key.size();
    while(i < line.size() && (line[i]==' ' || line[i]=='\t')) ++i;
    return i < line.size() && line[i] == ':';
}

static bool read_yaml_scalar(const std::string& path, const std::string& key, double& value)
{
    std::ifstream in(path.c_str());
    if(!in) return false;
    std::string line;
    while(std::getline(in,line))
    {
        if(!key_match(line,key)) continue;
        size_t c = line.find(':');
        if(c == std::string::npos) return false;
        try { value = std::stod(line.substr(c+1)); return true; }
        catch(...) { return false; }
    }
    return false;
}

static float physical_baseline_m(const rs2::device& dev)
{
    for(auto s : dev.query_sensors())
    {
        try
        {
            if(s.supports(RS2_OPTION_STEREO_BASELINE))
                return s.get_option(RS2_OPTION_STEREO_BASELINE) * 0.001f;
        }
        catch(...) {}
    }
    return -1.f;
}

static const char* distortion_name(rs2_distortion m)
{
    switch(m)
    {
        case RS2_DISTORTION_NONE: return "None";
        case RS2_DISTORTION_MODIFIED_BROWN_CONRADY: return "Modified Brown-Conrady";
        case RS2_DISTORTION_INVERSE_BROWN_CONRADY: return "Inverse Brown-Conrady";
        case RS2_DISTORTION_BROWN_CONRADY: return "Brown-Conrady";
        case RS2_DISTORTION_FTHETA: return "F-Theta";
        case RS2_DISTORTION_KANNALA_BRANDT4: return "Kannala-Brandt4";
        default: return "Unknown";
    }
}

static std::string sanitize_name(std::string s)
{
    for(char& c : s)
    {
        const bool ok =
            (c>='a'&&c<='z') || (c>='A'&&c<='Z') || (c>='0'&&c<='9') ||
            c=='-' || c=='_' || c=='.';
        if(!ok) c = '_';
    }
    if(s.empty()) s = "RealSense";
    return s;
}

static void make_dir_if_needed(const std::string& path)
{
    if(::mkdir(path.c_str(),0755) == 0 || errno == EEXIST) return;
    throw std::runtime_error("Cannot create directory '" + path + "': " + std::strerror(errno));
}

static std::string create_session_dir(const std::string& prefix)
{
    const std::string root = "trajectories";
    make_dir_if_needed(root);

    std::time_t now = std::time(NULL);
    std::tm tm_now;
    localtime_r(&now,&tm_now);
    char stamp[32];
    std::strftime(stamp,sizeof(stamp),"%Y%m%d_%H%M%S",&tm_now);

    const std::string base = root + "/" + sanitize_name(prefix) + "_" + stamp;
    for(int i=0;i<1000;++i)
    {
        const std::string p = (i==0) ? base : base + "_" + std::to_string(i);
        if(::mkdir(p.c_str(),0755) == 0) return p;
        if(errno != EEXIST)
            throw std::runtime_error("Cannot create session directory '" + p + "': " + std::strerror(errno));
    }
    throw std::runtime_error("Could not allocate a unique session directory");
}

static void write_runtime_yaml(const std::string& src, const std::string& dst,
                               const rs2_intrinsics& k, float depth_scale, int fps)
{
    std::ifstream in(src.c_str());
    if(!in) throw std::runtime_error("Cannot open YAML template: " + src);
    if(depth_scale <= 0.f) throw std::runtime_error("Invalid RealSense depth scale");

    std::map<std::string,std::string> v;
    v["Camera1.fx"] = num(k.fx); v["Camera1.fy"] = num(k.fy);
    v["Camera1.cx"] = num(k.ppx); v["Camera1.cy"] = num(k.ppy);
    v["Camera1.k1"] = "0.0"; v["Camera1.k2"] = "0.0";
    v["Camera1.p1"] = "0.0"; v["Camera1.p2"] = "0.0";
    v["Camera.width"] = std::to_string(k.width);
    v["Camera.height"] = std::to_string(k.height);
    v["Camera.fps"] = std::to_string(fps);
    v["Camera.RGB"] = "1";
    v["RGBD.DepthMapFactor"] = num(1.0/depth_scale,6);
    // Stereo.b is intentionally kept from the template.

    std::ofstream out(dst.c_str());
    if(!out) throw std::runtime_error("Cannot write runtime YAML: " + dst);

    std::string line;
    while(std::getline(in,line))
    {
        bool replaced=false;
        for(const auto& kv : v)
        {
            if(key_match(line,kv.first))
            {
                out << kv.first << ": " << kv.second << "\n";
                replaced=true;
                break;
            }
        }
        if(!replaced) out << line << "\n";
    }
}

static void make_maps(const rs2_intrinsics& k, cv::Mat& mx, cv::Mat& my, bool& use)
{
    bool zero=true;
    for(int i=0;i<5;++i) zero = zero && std::fabs(k.coeffs[i]) < 1e-10f;
    use = (k.model != RS2_DISTORTION_NONE && !zero);
    if(!use) return;

    mx.create(k.height,k.width,CV_32FC1);
    my.create(k.height,k.width,CV_32FC1);
    for(int y=0;y<k.height;++y)
    {
        float* px=mx.ptr<float>(y);
        float* py=my.ptr<float>(y);
        for(int x=0;x<k.width;++x)
        {
            float p[3]={(x-k.ppx)/k.fx,(y-k.ppy)/k.fy,1.f};
            float q[2];
            rs2_project_point_to_pixel(q,&k,p);
            px[x]=q[0]; py[x]=q[1];
        }
    }
}

static const char* tracking_state_name(int state)
{
    switch(state)
    {
        case ORB_SLAM3::Tracking::SYSTEM_NOT_READY: return "SYSTEM_NOT_READY";
        case ORB_SLAM3::Tracking::NO_IMAGES_YET: return "NO_IMAGES_YET";
        case ORB_SLAM3::Tracking::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case ORB_SLAM3::Tracking::OK: return "OK";
        case ORB_SLAM3::Tracking::RECENTLY_LOST: return "RECENTLY_LOST";
        case ORB_SLAM3::Tracking::LOST: return "LOST";
        case ORB_SLAM3::Tracking::OK_KLT: return "OK_KLT";
        default: return "UNKNOWN";
    }
}

struct PoseData
{
    double x=0,y=0,z=0;
    double qx=0,qy=0,qz=0,qw=1;
    double roll_deg=0,pitch_deg=0,yaw_deg=0;
};

static PoseData pose_from_Tcw(const Sophus::SE3f& Tcw)
{
    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Vector3f t = Twc.translation();
    Eigen::Quaternionf q = Twc.unit_quaternion();
    q.normalize();
    const Eigen::Matrix3f R = q.toRotationMatrix();

    const double roll = std::atan2((double)R(2,1),(double)R(2,2));
    const double pitch = std::atan2(-(double)R(2,0),
        std::sqrt((double)R(0,0)*R(0,0) + (double)R(1,0)*R(1,0)));
    const double yaw = std::atan2((double)R(1,0),(double)R(0,0));
    const double rad2deg = 180.0/3.14159265358979323846;

    PoseData p;
    p.x=t.x(); p.y=t.y(); p.z=t.z();
    p.qx=q.x(); p.qy=q.y(); p.qz=q.z(); p.qw=q.w();
    p.roll_deg=roll*rad2deg; p.pitch_deg=pitch*rad2deg; p.yaw_deg=yaw*rad2deg;
    return p;
}

static void write_live_row(std::ofstream& f, uint64_t frame_number, double timestamp_s,
                           int state, bool valid, const PoseData& p)
{
    f << std::fixed << frame_number << "," << std::setprecision(9) << timestamp_s << ","
      << tracking_state_name(state) << "," << (valid?1:0) << ",";
    if(valid)
    {
        f << p.x << "," << p.y << "," << p.z << ","
          << p.qx << "," << p.qy << "," << p.qz << "," << p.qw << ","
          << p.roll_deg << "," << p.pitch_deg << "," << p.yaw_deg;
    }
    else
    {
        f << "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan";
    }
    f << "\n";
    f.flush();
}

static void quaternion_to_rpy_deg(double qx,double qy,double qz,double qw,
                                  double& roll,double& pitch,double& yaw)
{
    Eigen::Quaterniond q(qw,qx,qy,qz);
    q.normalize();
    const Eigen::Matrix3d R=q.toRotationMatrix();
    roll=std::atan2(R(2,1),R(2,2));
    pitch=std::atan2(-R(2,0),std::sqrt(R(0,0)*R(0,0)+R(1,0)*R(1,0)));
    yaw=std::atan2(R(1,0),R(0,0));
    const double k=180.0/3.14159265358979323846;
    roll*=k; pitch*=k; yaw*=k;
}

static bool tum_to_csv(const std::string& tum, const std::string& csv)
{
    std::ifstream in(tum.c_str());
    std::ofstream out(csv.c_str());
    if(!in || !out) return false;
    out << "timestamp_s,x_m,y_m,z_m,qx,qy,qz,qw,roll_deg,pitch_deg,yaw_deg\n";
    double ts,x,y,z,qx,qy,qz,qw;
    while(in >> ts >> x >> y >> z >> qx >> qy >> qz >> qw)
    {
        double r,p,yy;
        quaternion_to_rpy_deg(qx,qy,qz,qw,r,p,yy);
        out << std::fixed << std::setprecision(9)
            << ts << "," << x << "," << y << "," << z << ","
            << qx << "," << qy << "," << qz << "," << qw << ","
            << r << "," << p << "," << yy << "\n";
    }
    return true;
}


struct SegmentMeta
{
    int id = 0;
    std::string dir;
    std::string live_csv_path;
    std::string final_csv_path;
    std::string final_tum_path;
    std::string keyframe_tum_path;
    std::string info_path;
    double start_ts = 0.0;
    double end_ts = 0.0;
    uint64_t frames = 0;
    uint64_t valid_frames = 0;
};

static std::string make_segment_dir(const std::string& session_dir, int id)
{
    std::ostringstream ss;
    ss << session_dir << "/segment_" << std::setw(3) << std::setfill('0') << id;
    const std::string path = ss.str();
    make_dir_if_needed(path);
    return path;
}

static void write_segment_info(const SegmentMeta& s)
{
    std::ofstream f(s.info_path.c_str());
    if(!f) return;

    f << std::fixed << std::setprecision(9);
    f << "segment_id=" << s.id << "\n";
    f << "start_timestamp_s=" << s.start_ts << "\n";
    f << "end_timestamp_s=" << s.end_ts << "\n";
    f << "frames_recorded=" << s.frames << "\n";
    f << "valid_pose_frames=" << s.valid_frames << "\n";
    f << "live_csv=" << s.live_csv_path << "\n";
    f << "final_csv=" << s.final_csv_path << "\n";
    f << "final_tum=" << s.final_tum_path << "\n";
    f << "keyframe_tum=" << s.keyframe_tum_path << "\n";
}

static bool filter_tum_range(const std::string& input_path,
                             const std::string& output_path,
                             double start_ts,
                             double end_ts)
{
    std::ifstream in(input_path.c_str());
    if(!in) return false;

    std::ofstream out(output_path.c_str());
    if(!out) return false;

    std::string line;
    bool wrote_any = false;
    const double eps = 1e-6;

    while(std::getline(in, line))
    {
        if(line.empty() || line[0] == '#')
            continue;

        std::istringstream ss(line);
        double ts = 0.0;
        if(!(ss >> ts))
            continue;

        if(ts + eps < start_ts || ts - eps > end_ts)
            continue;

        out << line << "\n";
        wrote_any = true;
    }

    return wrote_any;
}

static void write_session_info(const std::string& path, const std::string& device_name,
                               const std::string& serial, const rs2_intrinsics& k,
                               float depth_scale, float physical_b,
                               double template_b, bool have_template_b,
                               const std::string& template_yaml,
                               const std::string& runtime_yaml)
{
    std::ofstream f(path.c_str());
    if(!f) return;
    f << "device_name=" << device_name << "\nserial=" << serial << "\n";
    f << "width=" << k.width << "\nheight=" << k.height << "\n";
    f << std::setprecision(12)
      << "fx=" << k.fx << "\nfy=" << k.fy << "\ncx=" << k.ppx << "\ncy=" << k.ppy << "\n";
    f << "distortion_model=" << distortion_name(k.model) << "\n";
    f << "distortion_coeffs=" << k.coeffs[0] << "," << k.coeffs[1] << ","
      << k.coeffs[2] << "," << k.coeffs[3] << "," << k.coeffs[4] << "\n";
    f << "depth_scale_m_per_unit=" << depth_scale << "\n";
    f << "physical_ir_baseline_m=" << physical_b << "\n";
    if(have_template_b) f << "orbslam_rgbd_stereo_b_m=" << template_b << "\n";
    f << "template_yaml=" << template_yaml << "\nruntime_yaml=" << runtime_yaml << "\n";
    f << "pose_convention=Twc_camera_pose_in_ORB_SLAM3_world\n";
    f << "rpy_convention=ZYX_yaw_pitch_roll_degrees_from_Rwc\n";
    f << "live_csv_note=flushed_each_frame_and_may_differ_from_final_after_BA_or_loop_closure\n";
    f << "final_csv_note=derived_from_ORB_SLAM3_SaveTrajectoryTUM_after_shutdown_lost_frames_omitted\n";
}

struct TimingStats
{
    uint64_t frames=0, gaps=0, last=0;
    bool have_last=false;
    double align_sum=0, remap_sum=0, track_sum=0;
    void add(double a,double r,double t,uint64_t n)
    {
        ++frames; align_sum+=a; remap_sum+=r; track_sum+=t;
        if(have_last && n>last+1) gaps += n-last-1;
        last=n; have_last=true;
    }
    void print() const
    {
        if(!frames) return;
        double inv=1.0/frames;
        double proc=(align_sum+remap_sum+track_sum)*inv;
        std::cout << "\nPerformance summary\n"
                  << " frames: " << frames << "\n"
                  << " camera frame gaps: " << gaps << "\n"
                  << std::fixed << std::setprecision(2)
                  << " align avg: " << align_sum*inv << " ms\n"
                  << " remap avg: " << remap_sum*inv << " ms\n"
                  << " TrackRGBD avg: " << track_sum*inv << " ms\n"
                  << " processing avg: " << proc << " ms (~"
                  << (proc>0?1000.0/proc:0.0) << " FPS compute throughput)\n";
    }
};

int main(int argc,char** argv)
{
    if(argc<3 || argc>4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " vocabulary template_yaml [session_prefix]\n";
        return 1;
    }

    std::signal(SIGINT,on_signal);
    std::signal(SIGTERM,on_signal);

    try
    {
        const std::string vocabulary = argv[1];
        const std::string template_yaml = argv[2];
        const std::string prefix_name = (argc==4) ? argv[3] : "RealSense";

        const std::string session_dir = create_session_dir(prefix_name);
        const std::string runtime_yaml = session_dir + "/runtime.yaml";
        const std::string info_path = session_dir + "/session_info.txt";
        const std::string full_tum_path = session_dir + "/_internal_full_trajectory_tum.txt";
        const std::string full_kf_tum_path = session_dir + "/_internal_full_keyframe_tum.txt";

        rs2::context ctx;
        auto devs = ctx.query_devices();
        if(devs.size()==0)
            throw std::runtime_error("No RealSense device connected");

        rs2::device dev = devs[0];
        const std::string name =
            dev.supports(RS2_CAMERA_INFO_NAME)
                ? dev.get_info(RS2_CAMERA_INFO_NAME)
                : "RealSense";
        const std::string serial =
            dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)
                ? dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
                : "unknown";

        std::cout << "Device: " << name
                  << "\nSerial: " << serial
                  << "\nSession directory: " << session_dir << "\n";

        const int W=640,H=480,FPS=30;

        rs2::pipeline pipe;
        // Callback-driven queue keeps RealSense acquisition alive even while
        // ORB-SLAM3 is busy loading the vocabulary. Capacity 2 is intentional
        // for low-latency RGB+Depth streaming; old frames are dropped.
        rs2::frame_queue rs_queue(2, true);
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR,W,H,RS2_FORMAT_RGB8,FPS);
        cfg.enable_stream(RS2_STREAM_DEPTH,W,H,RS2_FORMAT_Z16,FPS);
        auto profile = pipe.start(cfg, rs_queue);

        auto cp = profile.get_stream(RS2_STREAM_COLOR)
                         .as<rs2::video_stream_profile>();
        const rs2_intrinsics k = cp.get_intrinsics();

        auto ds = profile.get_device().first<rs2::depth_sensor>();
        const float depth_scale = ds.get_depth_scale();
        const float physical_b = physical_baseline_m(profile.get_device());

        double template_b = std::numeric_limits<double>::quiet_NaN();
        const bool have_template_b =
            read_yaml_scalar(template_yaml,"Stereo.b",template_b);

        write_runtime_yaml(template_yaml,runtime_yaml,k,depth_scale,FPS);
        write_session_info(info_path,name,serial,k,depth_scale,physical_b,
                           template_b,have_template_b,template_yaml,runtime_yaml);

        std::cout << "\nActive calibration\n"
                  << " fx/fy: " << k.fx << " / " << k.fy << "\n"
                  << " cx/cy: " << k.ppx << " / " << k.ppy << "\n"
                  << " size : " << k.width << " x " << k.height << "\n"
                  << " model: " << distortion_name(k.model) << "\n"
                  << " depth scale: " << depth_scale << " m/unit\n"
                  << " physical IR baseline: " << physical_b << " m\n";

        if(have_template_b)
            std::cout << " ORB-SLAM3 RGB-D Stereo.b: "
                      << template_b << " m (kept from template)\n";

        cv::Mat mx,my;
        bool remap=false;
        std::cout << "Building rectification map..." << std::flush;
        make_maps(k,mx,my,remap);
        std::cout << (remap ? " enabled\n" : " not needed\n");

        rs2::align align(RS2_STREAM_COLOR);

        ORB_SLAM3::System SLAM(
            vocabulary,
            runtime_yaml,
            ORB_SLAM3::System::RGBD,
            true
        );

        // Discard buffered frames accumulated while ORB vocabulary was loading;
        // then require a newly-arrived coherent RGB-D frameset from the SDK queue.
        {
            rs2::frame stale;
            while(rs_queue.poll_for_frame(&stale)) {}

            rs2::frame first_frame;
            if(!rs_queue.try_wait_for_frame(&first_frame, 5000))
                throw std::runtime_error(
                    "RealSense frame_queue produced no frame after ORB initialization"
                );

            auto first_frameset = first_frame.as<rs2::frameset>();
            if(!first_frameset ||
               !first_frameset.get_color_frame() ||
               !first_frameset.get_depth_frame())
                throw std::runtime_error(
                    "RealSense frame_queue received a non RGB-D frameset"
                );
        }

        std::cout << "\n[RealSense] callback frame_queue RGB-D handoff OK.\n";

        std::cout
            << "\nTracking started. SLAM is always running.\n"
            << "Use the Pangolin buttons:\n"
            << "  Start Save Trajectory  -> begin a new recording segment\n"
            << "  Stop Save Trajectory   -> stop that segment, SLAM keeps running\n"
            << "Realtime pose remains visible at ~5 Hz.\n"
            << "Press Ctrl+C in this terminal only when you want to exit the program.\n\n";

        TimingStats stats;

        uint64_t processed = 0;
        bool recording = false;
        int next_segment_id = 1;

        std::ofstream segment_csv;
        SegmentMeta current_segment;
        std::vector<SegmentMeta> completed_segments;

        auto start_segment = [&](double ts)
        {
            if(recording)
            {
                std::cout << "\n[REC] Already recording segment "
                          << current_segment.id << "\n";
                return;
            }

            current_segment = SegmentMeta();
            current_segment.id = next_segment_id++;
            current_segment.dir = make_segment_dir(session_dir,current_segment.id);
            current_segment.live_csv_path =
                current_segment.dir + "/camera_trajectory_live.csv";
            current_segment.final_csv_path =
                current_segment.dir + "/camera_trajectory_final.csv";
            current_segment.final_tum_path =
                current_segment.dir + "/camera_trajectory_final_tum.txt";
            current_segment.keyframe_tum_path =
                current_segment.dir + "/keyframe_trajectory_final_tum.txt";
            current_segment.info_path =
                current_segment.dir + "/segment_info.txt";
            current_segment.start_ts = ts;
            current_segment.end_ts = ts;

            segment_csv.open(current_segment.live_csv_path.c_str());
            if(!segment_csv)
                throw std::runtime_error(
                    "Cannot open segment CSV: " + current_segment.live_csv_path
                );

            segment_csv
                << "frame_number,timestamp_s,tracking_state,valid,"
                << "x_m,y_m,z_m,qx,qy,qz,qw,"
                << "roll_deg,pitch_deg,yaw_deg\n";
            segment_csv.flush();

            recording = true;
            SLAM.SetTrajectoryRecordingState(true);

            std::cout
                << "\n========== RECORDING START ==========\n"
                << "Segment: " << current_segment.id << "\n"
                << "Live trajectory:\n  "
                << current_segment.live_csv_path << "\n"
                << "=====================================\n";
        };

        auto stop_segment = [&]()
        {
            if(!recording)
            {
                std::cout << "\n[REC] No active trajectory recording.\n";
                SLAM.SetTrajectoryRecordingState(false);
                return;
            }

            segment_csv.flush();
            segment_csv.close();

            write_segment_info(current_segment);
            completed_segments.push_back(current_segment);

            std::cout
                << "\n========== RECORDING STOP ==========\n"
                << "Segment: " << current_segment.id << "\n"
                << "Frames: " << current_segment.frames
                << " (valid pose: " << current_segment.valid_frames << ")\n"
                << "Immediate live CSV saved:\n  "
                << current_segment.live_csv_path << "\n"
                << "Final optimized CSV will be generated at program exit.\n"
                << "====================================\n";

            recording = false;
            SLAM.SetTrajectoryRecordingState(false);
        };

        int rs_queue_timeout_streak = 0;

        while(running && !SLAM.isShutDown())
        {
            rs2::frame queued_frame;
            if(!rs_queue.try_wait_for_frame(&queued_frame, 1000))
            {
                ++rs_queue_timeout_streak;
                if(rs_queue_timeout_streak == 1 || rs_queue_timeout_streak % 5 == 0)
                    std::cerr << "\n[RealSense] frame_queue timeout: "
                              << rs_queue_timeout_streak << " second(s)\n";
                continue;
            }

            rs_queue_timeout_streak = 0;
            auto frames = queued_frame.as<rs2::frameset>();
            if(!frames)
                continue;

            const auto t1 = Clock::now();
            auto aligned = align.process(frames);
            const auto t2 = Clock::now();

            auto color = aligned.get_color_frame();
            auto depth = aligned.get_depth_frame();
            if(!color || !depth)
                continue;

            cv::Mat c0(
                cv::Size(k.width,k.height),
                CV_8UC3,
                const_cast<void*>(color.get_data()),
                cv::Mat::AUTO_STEP
            );

            cv::Mat d0(
                cv::Size(k.width,k.height),
                CV_16UC1,
                const_cast<void*>(depth.get_data()),
                cv::Mat::AUTO_STEP
            );

            cv::Mat c,d;
            if(remap)
            {
                cv::remap(c0,c,mx,my,cv::INTER_LINEAR,cv::BORDER_CONSTANT);
                cv::remap(d0,d,mx,my,cv::INTER_NEAREST,cv::BORDER_CONSTANT);
            }
            else
            {
                c=c0;
                d=d0;
            }

            const auto t3 = Clock::now();

            const double ts = color.get_timestamp()*1e-3;
            const Sophus::SE3f Tcw = SLAM.TrackRGBD(c,d,ts);
            const auto t4 = Clock::now();

            const int state = SLAM.GetTrackingState();
            const bool valid =
                state==ORB_SLAM3::Tracking::OK ||
                state==ORB_SLAM3::Tracking::OK_KLT;

            PoseData pose;
            if(valid)
                pose = pose_from_Tcw(Tcw);

            const uint64_t frame_number = color.get_frame_number();

            // Consume viewer button commands on the tracking/main thread.
            const int record_command =
                SLAM.ConsumeTrajectoryRecordCommand();

            if(record_command == ORB_SLAM3::System::TRAJECTORY_RECORD_START)
            {
                start_segment(ts);
            }
            else if(record_command == ORB_SLAM3::System::TRAJECTORY_RECORD_STOP)
            {
                // Stop before writing this frame, so the button defines the
                // end boundary cleanly at the previous recorded frame.
                stop_segment();
            }
            else if(record_command == ORB_SLAM3::System::TRAJECTORY_APP_STOP)
            {
                // Viewer Stop is handled on this main thread so Shutdown and
                // trajectory finalization happen exactly once.
                running = false;
            }

            if(recording)
            {
                write_live_row(
                    segment_csv,
                    frame_number,
                    ts,
                    state,
                    valid,
                    pose
                );

                current_segment.end_ts = ts;
                ++current_segment.frames;
                if(valid)
                    ++current_segment.valid_frames;
            }

            ++processed;

            stats.add(
                ms_between(t1,t2),
                ms_between(t2,t3),
                ms_between(t3,t4),
                frame_number
            );

            if(processed%6==0)
            {
                std::cout
                    << "\r[POSE] "
                    << (recording ? "[REC] " : "[IDLE]")
                    << " "
                    << std::setw(15)
                    << tracking_state_name(state);

                if(valid)
                {
                    std::cout
                        << std::fixed
                        << std::setprecision(3)
                        << " xyz[m]=("
                        << pose.x << ", "
                        << pose.y << ", "
                        << pose.z << ")"
                        << " rpy[deg]=("
                        << pose.roll_deg << ", "
                        << pose.pitch_deg << ", "
                        << pose.yaw_deg << ")";

                    if(recording)
                        std::cout
                            << " seg=" << current_segment.id
                            << " saved=" << current_segment.valid_frames;
                }
                else
                {
                    std::cout << " pose=INVALID";
                }

                std::cout << "          " << std::flush;
            }
        }

        std::cout << "\n";

        // If Ctrl+C happens while recording, preserve and close that segment.
        if(recording)
        {
            std::cout
                << "\nCtrl+C received while recording; "
                << "closing the active segment safely.\n";
            stop_segment();
        }

        pipe.stop();
        stats.print();

        if(!SLAM.isShutDown())
        {
            std::cout << "\nStopping ORB-SLAM3...\n";
            SLAM.Shutdown();
        }
        else
        {
            std::cout
                << "\nORB-SLAM3 already shut down; "
                << "skipping a second Shutdown().\n";
        }

        if(!completed_segments.empty())
        {
            // Save one optimized full-session trajectory, then cut each
            // user-selected recording interval by timestamp.
            SLAM.SaveTrajectoryTUM(full_tum_path);
            SLAM.SaveKeyFrameTrajectoryTUM(full_kf_tum_path);

            std::cout
                << "\nGenerating optimized trajectory files "
                << "for " << completed_segments.size()
                << " recorded segment(s)...\n";

            for(auto& seg : completed_segments)
            {
                const bool have_final =
                    filter_tum_range(
                        full_tum_path,
                        seg.final_tum_path,
                        seg.start_ts,
                        seg.end_ts
                    );

                filter_tum_range(
                    full_kf_tum_path,
                    seg.keyframe_tum_path,
                    seg.start_ts,
                    seg.end_ts
                );

                if(have_final)
                {
                    if(!tum_to_csv(seg.final_tum_path,seg.final_csv_path))
                    {
                        std::cerr
                            << "WARNING: failed to convert segment "
                            << seg.id << " final TUM to CSV.\n";
                    }
                }
                else
                {
                    std::cerr
                        << "WARNING: segment " << seg.id
                        << " contains no valid optimized trajectory samples.\n";
                }

                write_segment_info(seg);
            }
        }

        std::cout
            << "\n========== SESSION COMPLETE ==========\n"
            << "Root directory:\n  " << session_dir << "\n"
            << "Recorded segments: " << completed_segments.size() << "\n";

        for(const auto& seg : completed_segments)
        {
            std::cout
                << "\nSegment " << seg.id << ":\n"
                << "  Live / crash-resistant CSV:\n    "
                << seg.live_csv_path << "\n"
                << "  Final optimized CSV:\n    "
                << seg.final_csv_path << "\n"
                << "  Final optimized TUM:\n    "
                << seg.final_tum_path << "\n";
        }

        std::cout
            << "\nUse camera_trajectory_final.csv as the primary result.\n"
            << "======================================\n";

        return 0;
    }
    catch(const rs2::error& e)
    {
        std::cerr << "RealSense error: " << e.what() << "\n";
        return 2;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 3;
    }
}
