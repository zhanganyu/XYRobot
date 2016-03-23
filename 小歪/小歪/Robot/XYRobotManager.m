//
//  XYRobotManager.m
//  小歪
//
//  Created by reese on 16/3/21.
//  Copyright © 2016年 com.ifenduo. All rights reserved.
//

#import "XYRobotManager.h"

@implementation XYRobotManager

//静态c指针 神经网络对象
static struct fann *ann;

+ (instancetype)sharedManager {
    static XYRobotManager *_inst;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        _inst = [XYRobotManager new];
        
        //配置神经网络初始参数
        [_inst initConfig];
    });
    return _inst;
}

- (void)initConfig {
    //3层神经元
    _neuralLayerNumber = 3;
    
    //96个内部神经元
    _hiddenNeuralNumber = 96;
    
    //2个输入
    _inputNum = 2;
    
    //1个输出
    _outputNum = 1;
    
    //预期错误均方差
    _desiredError = 0.01;

    //训练数据保存路径
    _trainDataPath = [[XYRobotManager dataPath] stringByAppendingPathComponent:@"train.data"];
    
    //大脑保存路径
    _networkSavingPath = [[XYRobotManager dataPath] stringByAppendingPathComponent:@"brain.net"];
    
    
}

+ (NSString *)cachePath {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
    
    NSString *documentsDirectory = [paths objectAtIndex:0];
    
    NSString *cacheFolder = [documentsDirectory stringByAppendingPathComponent:@"XYRobot"];
    
    if (![[NSFileManager defaultManager] fileExistsAtPath:cacheFolder]) {
        //如果不存在该文件夹 新建
        [[NSFileManager defaultManager] createDirectoryAtPath:cacheFolder withIntermediateDirectories:YES attributes:nil error:nil];
    }
    return cacheFolder;
}

+ (NSString *)dataPath {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    
    NSString *documentsDirectory = [paths objectAtIndex:0];
    
    NSString *dataPath = [documentsDirectory stringByAppendingPathComponent:@"XYRobot"];
    
    if (![[NSFileManager defaultManager] fileExistsAtPath:dataPath]) {
        //如果不存在该文件夹 新建
        [[NSFileManager defaultManager] createDirectoryAtPath:dataPath withIntermediateDirectories:YES attributes:nil error:nil];
    }
    return dataPath;
}

- (void)createBrain {
    NSLog(@"初始化神经网络");
    
    //如果之前保存过,从文件读取
    if ([[NSFileManager defaultManager] fileExistsAtPath:_networkSavingPath]) {
        ann = fann_create_from_file([_networkSavingPath cStringUsingEncoding:NSUTF8StringEncoding]);
    } else {
    
        //创建标准神经网络,配置好对应的参数
        ann = fann_create_standard(_neuralLayerNumber,_inputNum,_hiddenNeuralNumber,_outputNum);
        //给所有选择器初始权重(-0.1~0.1间的随机值)
        fann_randomize_weights(ann, -1, 1);
        //设置中间层的激活坡度(0~1之间)
        fann_set_activation_steepness_hidden(ann, 1);
        //设置输出层的激活坡度(0~1直接)
        fann_set_activation_steepness_output(ann, 1);
        
        //设置激活函数(激活函数有很多种，具体每一种的效果还没去试)
        fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
        fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    }
    
    
}

- (void)saveBrain {
    fann_save(ann, [_networkSavingPath cStringUsingEncoding:NSUTF8StringEncoding]);
}

static void callback(unsigned int dataCount, unsigned int inputNum, unsigned int outputNum, fann_type * input, fann_type *output) {
    NSLog(@"训练完成");
}

//inputData和outputDatas是float[]类型的数组
- (void)trainInputDatas:(fann_type *)inputData outputDatas:(fann_type *)outputData dataCount:(int)dataCount {
    
    //设置训练算法
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    
    //设置终止条件为bit fail
    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
    
    //设置bit fail limit 即所有训练结果中与预期不符合的个数上限，超出这个上限会一直训练
    fann_set_bit_fail_limit(ann, 0.0f);
    
    //设置训练速度(0~1之间,0为不加速)
    fann_set_learning_momentum(ann, 0.4f);
    
    //由当前函数传递的输入生成一个c指针
    struct fann_train_data* trainData = fann_create_train_from_callback(dataCount, _inputNum, _outputNum, callback);
    
    trainData->input=&inputData;
    trainData->output=&outputData;

    //之前是否存储过训练数据,如果有,和现在的合并(merge)
    if ([[NSFileManager defaultManager] fileExistsAtPath:_trainDataPath]) {
        //已经存储好的训练数据
        struct fann_train_data* trainDataSaved = fann_read_train_from_file([_trainDataPath cStringUsingEncoding:NSUTF8StringEncoding]);
        //合并
        trainData = fann_merge_train_data(trainData, trainDataSaved);
    }
    
    //开始训练,最大次数3000次,输出日志间隔:每次都输出 如果不需要重复训练,预期均方差设置为-1即可
    fann_train_on_data(ann, trainData, 3000, 1, _desiredError);
    
    //保存训练数据
    fann_save_train(trainData, [_trainDataPath cStringUsingEncoding:NSUTF8StringEncoding]);
    
    NSLog(@"训练数据路径%@",_trainDataPath);
}

- (NSArray *)runInputDatas:(fann_type *)inputData {
    fann_type* output = fann_run(ann, inputData);
    
    NSMutableArray* array = [NSMutableArray arrayWithCapacity:_outputNum];
    for (int i = 0; i<_outputNum ; i++) {
        [array addObject:@(output[i])];
        
    }
    return array;
}
@end
