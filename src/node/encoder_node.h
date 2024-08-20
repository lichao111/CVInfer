#pragma once

#include "node/node_base.h"
#include "tools/timer.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

namespace cv_infer
{
struct OutCfg
{
    std::string                                  out_url;
    std::string                                  codec    = "h264_nvenc";
    int64_t                                      bit_rate = 8000000;
    int                                          width    = 1920;
    int                                          height   = 1080;
    int                                          fps      = 30;
    std::unordered_map<std::string, std::string> opt      = {
        {"preset", "medium"}, {"profile", "main"}, {"crf", "18"}};

    // owt param
    std::string channelID;
    std::string appID;
    std::string appKey;
    std::string userID;
    std::string userName;
    uint32_t    timeoutHours{24};
    bool        async_leave{true};
};
class EncoderNode : public NodeBase
{
public:
    EncoderNode() : NodeBase(1, 0, 1) {}
    EncoderNode(const std::string &file_name)
        : NodeBase(1, 0, 1),
          OutFile(file_name){

          };
    virtual ~EncoderNode();
    bool                               Init(const std::string &file_name);
    virtual bool                       Run() override;
    virtual std::vector<SignalBasePtr> Worker(
        std::vector<SignalBasePtr> input_signals) override
    {
        return {};
    };

private:
    bool Open(const OutCfg &cfg);
    bool Close();
    bool PushOneFrame(const cv::Mat &frame);

    bool IsReady    = false;
    bool NeedTailer = false;
    // AVCtx
    AVFormatContext *Ctx      = nullptr;
    AVStream        *St       = nullptr;
    AVCodecContext  *CCtx     = nullptr;
    AVPacket        *Pkt      = nullptr;
    AVFrame         *Frame    = nullptr;
    SwsContext      *Sws      = nullptr;
    int              OutFlags = SWS_BILINEAR;
    std::string      OutFile;

    bool  FlushingEncodec();
    Timer CostTimer{"encoder"};
};
}  // namespace cv_infer