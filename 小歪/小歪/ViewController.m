//
//  ViewController.m
//  小歪
//
//  Created by reese on 16/3/21.
//  Copyright © 2016年 com.ifenduo. All rights reserved.
//

#import "ViewController.h"
#import "XYRobotManager.h"


@interface ViewController ()

@property (weak, nonatomic) IBOutlet UITextField *input1;
@property (weak, nonatomic) IBOutlet UITextField *input2;
@property (weak, nonatomic) IBOutlet UITextField *output1;
@property (weak, nonatomic) IBOutlet UILabel *output;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    //创建大脑
    [[XYRobotManager sharedManager] createBrain];
    
}

- (IBAction)train:(id)sender {
    fann_type input[2] = {_input1.text.floatValue,_input2.text.floatValue};
    fann_type output[1] = {_output1.text.floatValue};
    
    [[XYRobotManager sharedManager] trainInputDatas:input outputDatas:output dataCount:1];
    
}
- (IBAction)test:(id)sender {
    fann_type input[2] = {_input1.text.floatValue,_input2.text.floatValue};
    NSArray *output = [[XYRobotManager sharedManager] runInputDatas:input];
    [_output setText:[output.firstObject stringValue]];
}

@end
