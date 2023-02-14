#include  "plate_rec.h"


// #define CHECK(status) \
//     do\
//     {\
//         auto ret = (status);\
//         if (ret != 0)\
//         {\
//             std::cerr << "Cuda failure: " << ret << std::endl;\
//             abort();\
//         }\
//     } while (0)


void blobFromImage_plate(float *blob,cv::Mat& img,float mean_value,float std_value)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    int k = 0;
    for (size_t c = 0; c <3; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                    blob[k++] =
                    ((float)img.at<cv::Vec3b>(h, w)[c]/255.0-mean_value)/std_value;
            }
        }
    }
    // return blob;
}

void decode_outputs(float *prob,int output_size,int time_step,int num_char,std::string &plate_no)
{
    int pre_index = 0;
    for(int i = 0; i<time_step;i++)
    {
        float *row_ptr = prob+num_char*i;
        int max_value = row_ptr[0];
        int max_index = 0;
        for(int j = 1; j<num_char;j++)
        {
            if (row_ptr[j]>max_value)
            {
                max_value= row_ptr[j];
                max_index = j;
            }
        }
        if (max_index!=0 && max_index!=pre_index)
        {
            plate_no+=plate_string[max_index];
        }
        pre_index= max_index;
    }
}


PlateRec::PlateRec()
{
    this->prob_1 = new float[output_size];
    this->prob_2 = new float[output_size_color];
    this->blob = new float[3*168*48];
  
}

void PlateRec::loadTrtModel(const char *trtmodel,Logger &gLogger)
{
       char *trtModelStream{nullptr};
       size_t size{0};
       const std::string engine_file_path {trtmodel};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    context = engine->createExecutionContext();
    assert(context != nullptr);

    std::cout<<"loda plate_rec engine success!"<<std::endl;
    delete[] trtModelStream; 
}


void doInference(IExecutionContext& context, float* input, float* output,float *output_color,
 const int output_size,int output_size_color, cv::Size input_shape,const char *INPUT_BLOB_NAME,const char *OUTPUT_BLOB_NAME,const char * OUTPUT_BLOB_NAME_COLOR) 
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

     const int outputIndex_c = engine.getBindingIndex(OUTPUT_BLOB_NAME_COLOR);
    assert(engine.getBindingDataType(outputIndex_c) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_c], output_size_color*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    // context.enqueue(1, buffers, stream, nullptr);
    context.enqueueV2( buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_color, buffers[outputIndex_c], output_size_color * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(buffers[outputIndex_c]));
}


void PlateRec::plate_rec_color(cv::Mat &img,cv::Size size,std::string &plate_no,std::string &plate_color)
{
        int img_w = img.cols;
        int img_h = img.rows;
        blobFromImage_plate(blob,img,mean_value,std_value);
        doInference(*context, blob, prob_1,prob_2, output_size,output_size_color, img.size(),plate_rec_input_name,plate_rec_out_name_1,plate_rec_out_name_2);

        float max = prob_2[0];
        int color_index = 0;
        for (int i = 1 ; i<output_size_color;i++)
        {
               if(prob_2[i]>max)
               {
                max = prob_2[i];
                color_index = i;
               }
        }
        plate_color = color_list[color_index];

        decode_outputs(prob_1,output_size,time_step,num_char,plate_no);
}

PlateRec::~PlateRec()
{
    if (context)
    context->destroy();
    if (engine)
    engine->destroy();
    if(runtime)
    runtime->destroy();
    if(prob_1)
    delete [] prob_1;
    if(prob_2)
    delete [] prob_2;
    if(blob)
    delete [] blob;
}