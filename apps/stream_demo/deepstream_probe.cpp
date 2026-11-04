// Headless, full-frame classification. TensorRT executes inside nvinfer.
// Timing starts at decoded GPU-buffer arrival, NOT at camera exposure.
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <iterator>

using Clock = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double,std::milli>(Clock::now().time_since_epoch()).count(); }
static const std::array<const char*,5> class_names={"good","bent","loose","missing","pierced"};
struct Prediction { guint64 frame_num=0; int class_index=0; std::array<float,5> probabilities{}; };
struct State {
  GMainLoop* loop = nullptr;
  GstElement* mux = nullptr;
  std::mutex mutex;
  std::map<guint64,double> ingress;
  std::map<guint64,Prediction> preview_predictions;
  Prediction latest_prediction;
  bool has_prediction=false;
  std::ofstream output;
  std::filesystem::path preview_dir;
  std::string preview_prefix;
  guint64 preview_sequence=0;
  guint64 decoded=0, predicted=0, unmatched=0, invalid=0;
  guint64 max_predictions=0;
  double first=0, last=0, startup=now_ms(), seconds=0;
  std::atomic<bool> failed{false};
  bool eos=false;
  double decision_threshold=0;
};

static int choose_class(const std::array<float,5>& values,double threshold) {
  if (threshold>0) {
    if (1.0-static_cast<double>(values[0])<threshold) return 0;
    return static_cast<int>(std::max_element(values.begin()+1,values.end())-values.begin());
  }
  return static_cast<int>(std::max_element(values.begin(),values.end())-values.begin());
}

static void atomic_write(const std::filesystem::path& path, const char* data, size_t size) {
  auto temporary=path; temporary += ".tmp";
  { std::ofstream file(temporary,std::ios::binary|std::ios::trunc); file.write(data,static_cast<std::streamsize>(size)); file.flush();
    if (!file) throw std::runtime_error("Cannot write preview artifact: "+temporary.string()); }
  if (std::rename(temporary.string().c_str(),path.string().c_str())!=0)
    throw std::runtime_error("Cannot publish preview artifact "+path.string()+": "+std::strerror(errno));
}

static GstFlowReturn preview_sample(GstAppSink* sink, gpointer data) {
  auto& s=*static_cast<State*>(data);
  GstSample* sample=gst_app_sink_pull_sample(sink);
  if (!sample) return GST_FLOW_EOS;
  GstBuffer* buffer=gst_sample_get_buffer(sample);
  GstMapInfo map=GST_MAP_INFO_INIT;
  if (!buffer || !gst_buffer_map(buffer,&map,GST_MAP_READ)) { gst_sample_unref(sample); return GST_FLOW_ERROR; }
  try {
    Prediction prediction;
    bool matched=false;
    bool latest_completed=false;
    const guint64 pts=GST_BUFFER_PTS(buffer);
    guint64 sequence=0;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      auto& predictions=s.preview_predictions;
      auto found=predictions.lower_bound(pts);
      auto best=predictions.end();
      guint64 best_delta=G_MAXUINT64;
      if (found!=predictions.end()) { best=found; best_delta=found->first>pts?found->first-pts:pts-found->first; }
      if (found!=predictions.begin()) {
        auto previous=std::prev(found); guint64 delta=previous->first>pts?previous->first-pts:pts-previous->first;
        if (delta<best_delta) { best=previous; best_delta=delta; }
      }
      // Decode/encoder stages can shift RTSP PTS by a few milliseconds; match only
      // within half a 30-FPS frame so a neighbouring prediction is never guessed.
      if (best!=predictions.end() && best_delta<=15'000'000ULL) {
        prediction=best->second; matched=true; predictions.erase(predictions.begin(),std::next(best));
      } else if (s.has_prediction) {
        prediction=s.latest_prediction; latest_completed=true;
      }
      sequence=++s.preview_sequence;
    }
    std::ostringstream suffix; suffix << "frame_" << s.preview_prefix << "_" << std::setw(10) << std::setfill('0') << sequence;
    const std::string frame_file=suffix.str()+".jpg";
    atomic_write(s.preview_dir/frame_file,reinterpret_cast<const char*>(map.data),map.size);
    std::ostringstream state;
    state.precision(9);
    const bool prediction_available=matched||latest_completed;
    state << "{\"ready\":true,\"prediction_available\":" << (prediction_available?"true":"false")
      << ",\"pts_ns\":" << pts << ",\"frame_file\":\"" << frame_file << "\"";
    if (prediction_available) {
      state << ",\"association\":\"" << (matched?"pts":"latest_completed") << "\",\"prediction_frame_num\":" << prediction.frame_num
        << ",\"class\":\"" << class_names[prediction.class_index]
        << "\",\"defective\":" << (prediction.class_index==0?"false":"true") << ",\"confidence\":"
        << prediction.probabilities[prediction.class_index] << ",\"probabilities\":{";
      for (int i=0;i<5;++i) state << (i?",":"") << "\"" << class_names[i] << "\":" << prediction.probabilities[i];
      state << "}";
    }
    state << "}\n";
    const std::string text=state.str();
    std::string state_file=suffix.str()+".json";
    atomic_write(s.preview_dir/state_file,text.data(),text.size());
    if (sequence>60) {
      std::ostringstream stale; stale << "frame_" << s.preview_prefix << "_" << std::setw(10) << std::setfill('0') << (sequence-60);
      std::remove((s.preview_dir/(stale.str()+".jpg")).string().c_str());
      std::remove((s.preview_dir/(stale.str()+".json")).string().c_str());
    }
  } catch (const std::exception& error) {
    std::cerr << "Preview output error: " << error.what() << "\n";
    s.failed=true;
  }
  gst_buffer_unmap(buffer,&map);
  gst_sample_unref(sample);
  return s.failed?GST_FLOW_ERROR:GST_FLOW_OK;
}

static GstPadProbeReturn incoming(GstPad*, GstPadProbeInfo* info, gpointer data) {
  auto& s = *static_cast<State*>(data);
  auto* b = GST_PAD_PROBE_INFO_BUFFER(info);
  if (b) {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.ingress[GST_BUFFER_PTS(b)] = now_ms();
    ++s.decoded;
  }
  return GST_PAD_PROBE_OK;
}

static void pad_added(GstElement*, GstPad* pad, gpointer data) {
  auto& s = *static_cast<State*>(data);
  GstCaps* caps = gst_pad_get_current_caps(pad);
  if (!caps) caps = gst_pad_query_caps(pad, nullptr);
  bool video = caps && g_str_has_prefix(gst_structure_get_name(gst_caps_get_structure(caps,0)), "video/");
  bool nvmm = caps && gst_caps_features_contains(gst_caps_get_features(caps,0), "memory:NVMM");
  if (caps) gst_caps_unref(caps);
  if (!video) return;
  if (!nvmm) { std::cerr << "Source did not produce GPU NVMM video; refusing CPU fallback\n"; s.failed=true; g_main_loop_quit(s.loop); return; }
  GstPad* sink = gst_element_get_static_pad(s.mux, "sink_0");
  if (gst_pad_link(pad,sink) != GST_PAD_LINK_OK) { s.failed=true; g_main_loop_quit(s.loop); }
  gst_object_unref(sink);
}

static GstPadProbeReturn predicted(GstPad*, GstPadProbeInfo* info, gpointer data) {
  auto& s = *static_cast<State*>(data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) return GST_PAD_PROBE_OK;
  for (auto* l=batch->frame_meta_list; l; l=l->next) {
    auto* frame=static_cast<NvDsFrameMeta*>(l->data);
    for (auto* u=frame->frame_user_meta_list; u; u=u->next) {
      auto* user=static_cast<NvDsUserMeta*>(u->data);
      if (user->base_meta.meta_type!=NVDSINFER_TENSOR_OUTPUT_META) continue;
      auto* meta=static_cast<NvDsInferTensorMeta*>(user->user_meta_data);
      for (unsigned int k=0;k<meta->num_output_layers;++k) {
        auto& layer=meta->output_layers_info[k];
        if (std::string(layer.layerName)!="probabilities") continue;
        if (layer.dataType!=FLOAT || layer.inferDims.numElements!=5 || !meta->out_buf_ptrs_host[k]) {
          ++s.invalid; s.failed=true; g_main_loop_quit(s.loop); continue;
        }
        const float* p=static_cast<const float*>(meta->out_buf_ptrs_host[k]);
        std::array<float,5> values;
        std::copy(p,p+5,values.begin());
        float sum=0;
        for (float v:values) { if (!std::isfinite(v) || v<0 || v>1.001f) s.failed=true; sum+=v; }
        if (std::abs(sum-1)>0.01f || s.failed) { ++s.invalid; s.failed=true; g_main_loop_quit(s.loop); continue; }
        double done=now_ms(), elapsed=-1;
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it=s.ingress.find(frame->buf_pts);
        if (it!=s.ingress.end()) { elapsed=done-it->second; s.ingress.erase(it); }
        else ++s.unmatched;
        if (!s.first) s.first=done;
        s.last=done;
        ++s.predicted;
        if (!s.preview_dir.empty()) {
          Prediction record; record.frame_num=frame->frame_num; record.probabilities=values;
          record.class_index=choose_class(values,s.decision_threshold);
          s.latest_prediction=record; s.has_prediction=true;
          s.preview_predictions[frame->buf_pts]=record;
          while (s.preview_predictions.size()>64) s.preview_predictions.erase(s.preview_predictions.begin());
        }
        s.output << "{\"frame_num\":" << frame->frame_num << ",\"pts_ns\":" << frame->buf_pts
          << ",\"source_id\":" << frame->source_id << ",\"completed_steady_ms\":" << std::fixed << done
          << ",\"decode_output_to_prediction_ms\":" << elapsed
          << ",\"class_index\":" << choose_class(values,s.decision_threshold) << ",\"probabilities\":[";
        for (int i=0;i<5;++i) s.output << (i?",":"") << values[i];
        s.output << "]}\n";
        // The browser polls this file while inference is running; publish each
        // completed image immediately instead of waiting for EOS/stream stop.
        s.output.flush();
        if (s.max_predictions && s.predicted>=s.max_predictions) g_main_loop_quit(s.loop);
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

static gboolean bus_message(GstBus*, GstMessage* message, gpointer data) {
  auto& s=*static_cast<State*>(data);
  if (GST_MESSAGE_TYPE(message)==GST_MESSAGE_ERROR) {
    GError* error=nullptr; gchar* debug=nullptr;
    gst_message_parse_error(message,&error,&debug);
    std::cerr << "GST ERROR: " << error->message << "\n" << (debug?debug:"") << "\n";
    g_clear_error(&error); g_free(debug); s.failed=true; g_main_loop_quit(s.loop);
  } else if (GST_MESSAGE_TYPE(message)==GST_MESSAGE_EOS) { s.eos=true; g_main_loop_quit(s.loop); }
  return G_SOURCE_CONTINUE;
}

static gboolean deadline(gpointer data) {
  auto& s=*static_cast<State*>(data);
  std::lock_guard<std::mutex> lock(s.mutex);
  double now=now_ms();
  if (!s.first && now-s.startup>180000) { s.failed=true; std::cerr<<"No prediction within 180 seconds\n"; g_main_loop_quit(s.loop); }
  if (s.first && s.seconds>0 && now-s.first>=s.seconds*1000) g_main_loop_quit(s.loop);
  if (s.last && now-s.last>15000) { s.failed=true; std::cerr<<"Prediction stream stalled >15 seconds\n"; g_main_loop_quit(s.loop); }
  return G_SOURCE_CONTINUE;
}

int main(int argc,char** argv) {
  if (argc!=5 && argc!=6 && argc!=7) { std::cerr<<"Usage: deepstream_probe URI NvinferConfig OutputJsonl Seconds(0=EOS) [PreviewDirectory [MaxPredictions]]\n"; return 2; }
  gst_init(&argc,&argv);
  State s;
  if (const char* raw=std::getenv("STRAW_DECISION_THRESHOLD")) {
    if (*raw) {
      s.decision_threshold=std::stod(raw);
      if (!(s.decision_threshold>0 && s.decision_threshold<1)) throw std::runtime_error("Invalid stream decision threshold");
    }
  }
  s.seconds=std::stod(argv[4]); s.output.open(argv[3]);
  if (!s.output) return 2;
  if (argc>=6 && std::string(argv[5]).size()) {
    s.preview_dir=argv[5]; std::filesystem::create_directories(s.preview_dir);
    s.preview_prefix=std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
  }
  if (argc==7) s.max_predictions=std::stoull(argv[6]);
  s.output.precision(9);
  s.loop=g_main_loop_new(nullptr,FALSE);
  auto* pipeline=gst_pipeline_new("straw-stream-test");
  auto* source=gst_element_factory_make("nvurisrcbin","source");
  s.mux=gst_element_factory_make("nvstreammux","mux");
  auto* infer=gst_element_factory_make("nvinfer","infer");
  auto* sink=s.preview_dir.empty()?gst_element_factory_make("fakesink","sink"):nullptr;
  auto* queue=s.preview_dir.empty()?nullptr:gst_element_factory_make("queue","preview_queue");
  auto* convert=s.preview_dir.empty()?nullptr:gst_element_factory_make("nvvideoconvert","preview_convert");
  auto* capsfilter=s.preview_dir.empty()?nullptr:gst_element_factory_make("capsfilter","preview_caps");
  auto* jpeg=s.preview_dir.empty()?nullptr:gst_element_factory_make("jpegenc","preview_jpeg");
  auto* appsink=s.preview_dir.empty()?nullptr:gst_element_factory_make("appsink","preview_sink");
  if (!pipeline||!source||!s.mux||!infer||(s.preview_dir.empty()?(!sink):(!queue||!convert||!capsfilter||!jpeg||!appsink))) {
    std::cerr<<"Missing required DeepStream/GStreamer preview plugins\n"; return 2;
  }
  bool live=g_str_has_prefix(argv[1],"rtsp://") || g_str_has_prefix(argv[1],"rtsps://");
  g_object_set(source,"uri",argv[1],"disable-audio",TRUE,"gpu-id",0,"cudadec-memtype",0,nullptr);
  if (live) g_object_set(source,"latency",100,"select-rtp-protocol",4,"rtsp-reconnect-interval",5,"rtsp-reconnect-attempts",3,nullptr);
  g_object_set(s.mux,"batch-size",1,"width",640,"height",640,"live-source",live,"batched-push-timeout",4000,nullptr);
  g_object_set(infer,"config-file-path",argv[2],nullptr);
  gst_bin_add_many(GST_BIN(pipeline),source,s.mux,infer,nullptr);
  bool linked=false;
  if (s.preview_dir.empty()) {
    g_object_set(sink,"sync",FALSE,"async",FALSE,nullptr);
    gst_bin_add(GST_BIN(pipeline),sink);
    linked=gst_element_link_many(s.mux,infer,sink,nullptr);
  } else {
    GstCaps* caps=gst_caps_from_string("video/x-raw,format=I420");
    g_object_set(capsfilter,"caps",caps,nullptr); gst_caps_unref(caps);
    g_object_set(queue,"leaky",2,"max-size-buffers",1,"max-size-bytes",0,"max-size-time",static_cast<guint64>(0),nullptr);
    g_object_set(jpeg,"quality",78,nullptr);
    g_object_set(appsink,"sync",FALSE,"async",FALSE,"max-buffers",1,"drop",TRUE,"wait-on-eos",FALSE,nullptr);
    GstAppSinkCallbacks callbacks{}; callbacks.new_sample=preview_sample;
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink),&callbacks,&s,nullptr);
    gst_bin_add_many(GST_BIN(pipeline),queue,convert,capsfilter,jpeg,appsink,nullptr);
    linked=gst_element_link_many(s.mux,infer,queue,convert,capsfilter,jpeg,appsink,nullptr);
  }
  if (!linked) { std::cerr<<"Could not link DeepStream inference/preview pipeline\n"; return 2; }
  GstPad* request=gst_element_request_pad_simple(s.mux,"sink_0");
  gst_pad_add_probe(request,GST_PAD_PROBE_TYPE_BUFFER,incoming,&s,nullptr);
  gst_object_unref(request);
  g_signal_connect(source,"pad-added",G_CALLBACK(pad_added),&s);
  GstPad* out=gst_element_get_static_pad(infer,"src");
  gst_pad_add_probe(out,GST_PAD_PROBE_TYPE_BUFFER,predicted,&s,nullptr); gst_object_unref(out);
  GstBus* bus=gst_element_get_bus(pipeline);
  guint watch=gst_bus_add_watch(bus,bus_message,&s); gst_object_unref(bus);
  guint timer=g_timeout_add(100,deadline,&s);
  auto result=gst_element_set_state(pipeline,GST_STATE_PLAYING);
  if (!s.preview_dir.empty()) {
    std::ofstream marker(s.preview_dir/"session_running",std::ios::trunc); marker << "running\n"; marker.close();
  }
  if (result==GST_STATE_CHANGE_FAILURE) s.failed=true;
  else g_main_loop_run(s.loop);
  gst_element_set_state(pipeline,GST_STATE_NULL);
  if (!s.preview_dir.empty()) std::remove((s.preview_dir/"session_running").string().c_str());
  g_source_remove(timer); g_source_remove(watch);
  s.output.close();
  std::ofstream summary(std::string(argv[3])+".summary.json");
  summary << "{\"decoded\":"<<s.decoded<<",\"predicted\":"<<s.predicted<<",\"unmatched_pts\":"<<s.unmatched
    <<",\"invalid_tensors\":"<<s.invalid<<",\"pending_at_stop\":"<<s.ingress.size()<<",\"failed\":"<<(s.failed?"true":"false")
    <<",\"eos\":"<<(s.eos?"true":"false")<<",\"first_prediction_startup_ms\":"<<(s.first?s.first-s.startup:-1)
    <<",\"observed_seconds\":"<<(s.predicted>1?(s.last-s.first)/1000:0)<<"}\n";
  gst_object_unref(pipeline); g_main_loop_unref(s.loop);
  return s.failed||s.predicted==0?3:0;
}
