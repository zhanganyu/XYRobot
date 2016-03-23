//
//  XYRobotManager.h
//  小歪
//
//  Created by reese on 16/3/21.
//  Copyright © 2016年 com.ifenduo. All rights reserved.
//

#import <Foundation/Foundation.h>
#include "fann.h"

@interface XYRobotManager : NSObject


//神经网络层数
@property (nonatomic) int neuralLayerNumber;

//隐藏神经元个数 (中间层)
@property (nonatomic) int hiddenNeuralNumber;

//输入原件个数
@property (nonatomic) int inputNum;

//输出原件个数
@property (nonatomic) int outputNum;

//预期错误均方差
@property (nonatomic) float desiredError;

//训练数据存储路径
@property (nonatomic) NSString* trainDataPath;

//神经网络保存路径
@property (nonatomic) NSString* networkSavingPath;

//单例获取
+ (instancetype)sharedManager;

//创建大脑
- (void)createBrain;

//保存大脑
- (void)saveBrain;

//训练
- (void)trainInputDatas:(fann_type *)inputData outputDatas:(fann_type *)outputData dataCount:(int)dataCount;

//执行
- (NSArray *)runInputDatas:(fann_type *)inputData;
@end
