# coding=utf-8
import os
import io_file
import utils
from operator import itemgetter
import matplotlib.pyplot as plt


class evaluation(object):

    def __init__(self, cfg):
        # 参数初始化
        self.path = cfg['file_dir']
        self.overlapRatio = cfg['overlapRatio']
        self.cls = cfg['cls']
        self.precision = cfg['precision']
        self.recall = cfg['recall']
        self.threshold = cfg['threshold']
        self.FPPIW = cfg['FPPIW']
        self.roc = cfg['roc']
        self.pr = cfg['pr']
        self.tp = []
        self.fp = []
        self.all_num_pos = 0
        self.num_imgs = 0

    def load_all_files(self):
        # 加载数据类型和路径信息
        prediction_path = self.path[0]   # 预测结果
        gt_path = self.path[1]   # 真实结果
        if not os.path.exists(prediction_path):
            print('Incorrect detection results path! Please check and retry ~')
            return 0
        if not os.path.exists(gt_path):
            print('Incorrect ground truth path! Please check and retry ~')
            return 0
        if self.overlapRatio < 0 or self.overlapRatio > 1:
            print('Incorrect overlapRatio value! It should be in [0, 1]. Please check and retry ~')
            return 0
        if self.threshold < 0 or self.threshold > 1:
            print('Incorrect threshold value! It should be in [0, 1]. Please check and retry ~')
            return 0
        # 判断图片对是否合理
        pre_files = os.listdir(prediction_path)
        print ("Num of prediction files: ", len(pre_files))
        gt_files = os.listdir(gt_path)
        print ("Num of ground truth files: ", len(gt_files))
        if len(pre_files) != len(gt_files):
            print("groundtruths' size does not match predictions' size， please check ~ ")
            return 0
        elif len(pre_files) < 1:
            print('No files! Please check~')
            return 0
        predictions = []
        groundtruths = []
        file_format = ['.txt'] * 2  # [预测的格式， gt的格式]
        self.num_imgs = len(pre_files)
        for idx, item in enumerate(pre_files):
            if idx == 0:
                # 通过第一个文件判断当前文件的格式是否是XML
                if item.find('xml') >= 0:
                    file_format[0] = '.xml'
                    print('Got prediction file input in XML type ')
                elif item.find('txt'):
                    file_format[0] = '.txt'
                    print('Got prediction file input in TXT type ')
                else:
                    print('Unknown type!')
                    return 0
            predictions.append(os.path.join(item))  # 每张预测图片的路径
        for idx, item in enumerate(gt_files):
            if idx == 0:
                # 通过第一个文件判断当前文件的格式是否是XML
                if item.find('xml') >= 0:
                    file_format[1] = '.xml'
                    print('Got ground truth file input in XML type ')
                elif item.find('txt'):
                    file_format[1] = '.txt'
                    print('Got ground truth file input in TXT type ')
                else:
                    print('Unknown type!')
                    return 0
            groundtruths.append(os.path.join(item))  # 每张真实图片的路经
        # 返回[预测结果文件夹，真实数据文件夹，预测图片路经，真实图片路经路经，文件格式]
        return prediction_path, gt_path, predictions, groundtruths, file_format

    def cumTpFp(self, gtFile, detFile, label, overlapRatio, file_format):
        # 计算对应的图片中预测结果的TP和FP
        gtRects = []  # gtRect: label, xmin, ymin, xmax, ymax
        detRects = []  # detRect: label, xmin, ymin, xmax, ymax, score
        scores = []  # 当前类别的置信度
        num_pos = 0  # 真实数据中目标的个数
        if file_format[1] == '.txt':
            gtCon = open(gtFile)  # 打开当前文件
            gtLines = gtCon.readlines()  # 读取内容
            for gtLine in gtLines:
                if gtLine.split(' ')[0] == str(label):
                    gtRects.append((int(float(gtLine.split(' ')[1])), int(float(gtLine.split(' ')[2])),
                                    int(float(gtLine.split(' ')[3])), int(float(gtLine.split(' ')[4].strip('\n')))))
                    num_pos += 1
        elif file_format[1] == '.xml':
            gtLines=[]
            rects_xml = io_file.parse_xml(gtFile)  # 获取xml文件内容
            for rect_xml in rects_xml:
                if rect_xml[0] == str(label):
                    gtRects.append((rect_xml[1], rect_xml[2], rect_xml[3], rect_xml[4]))
                    gtLines.append(0)
                    num_pos += 1

        detCon = open(detFile)
        detLines = detCon.readlines()
        for detLine in detLines:
            # 统计当前类别下的目标
            if detLine.split(' ')[0] == str(label):
                detRects.append((int(detLine.split(' ')[1]), int(detLine.split(' ')[2]),
                                 int(detLine.split(' ')[3]), int(detLine.split(' ')[4])))
                scores.append(float(detLine.split(' ')[5].strip('\n')))
        # 统计当前类别下的TP和FP
        # det_state: [label, score, tp, fp], tp, fp = 0 or 1
        det_state = [(label, 0., 0, 1)] * len(detRects)
        iou_max = 0
        maxIndex = -1
        blockIdx = -1
        for cnt in range(len(det_state)):
            det_state[cnt] = (label, scores[cnt], 0, 1)  # 更新score值
        visited = [0] * len(gtLines)
        if len(detRects) != len(scores):
            print("Num of scores does not match detection results!")

        for indexDet, deti in enumerate(detRects):
            # 双循环保证乱序下的正确统计
            iou_max = 0
            maxIndex = -1
            blockIdx = -1
            for indexGt, gti in enumerate(gtRects):
                iou = utils.JaccardOverlap(detRects[indexDet], gtRects[indexGt])
                if iou > iou_max:
                    iou_max = iou
                    maxIndex = indexDet
                    blockIdx = indexGt
            # 没有进行NMS抑制，所以不使用标记位
            if iou_max >= overlapRatio and visited[blockIdx] == 0:
                det_state[maxIndex] = (label, scores[indexDet], 1, 0)
                visited[blockIdx] = 1
        # 返回每个预测结果的TP和FP统计，以及真实数据中目标数量
        return det_state, num_pos

    def get_tp_fp(self, gt_path, prediction_path, groundtruths, predictions, label, file_format):
        # 计算所有图片的TP和FP
        state_all = []
        self.tp = []  # tp = [(float, int)]
        self.fp = []  # fp = [(float, int)]
        self.all_num_pos = 0  # 所有目标的个数
        for groundtruth in groundtruths:
            name = groundtruth.strip(file_format[1])  # 获取真实每张图片的名称
            prediction = name + file_format[0]   # 获取响应预测图片的名称
            if prediction not in predictions:
                print(groundtruth, ': can not find corresponding file in prediction!')
                return 0, 0, 0
            groundtruth = os.path.join(gt_path, groundtruth)  # 真实图片数据的路经
            prediction = os.path.join(prediction_path, prediction)  # 预测图片结果路经

            det_state, num_pos = self.cumTpFp(groundtruth, prediction, label, self.overlapRatio, file_format)
            self.all_num_pos += num_pos  # 总共目标数
            state_all += det_state   # list.append操作

        for state in state_all:
            # print(state_all)
            self.tp.append((state[1], state[2]))
            self.fp.append((state[1], state[3]))
        return 0

    def CumSum(self):
        #计算器 FP，并返回错负个数，以及发段累计数据
        # 按照Score值排序
        fp_copy = sorted(self.fp, key=itemgetter(0), reverse=True)  # itemgetter函数用于获取对象的哪些维的数据
        cumsum = []
        cumPre = 0
        fp_th = 0
        fp_th_num = 0
        for index, pair in enumerate(fp_copy):
            cumPre += (fp_copy[index][1])
            cumsum.append(cumPre)  # 分短累计
            if fp_copy[index][0] > self.threshold:
                fp_th_num += 1
                if fp_copy[index][1] == 1:  # false positive
                    fp_th += 1
        fppw = float(fp_th) / float(fp_th_num)  # FP
        return cumsum, fp_th, fppw

    def CumSum_tp(self):
        # 计算TP，并返回正确分类的个数和累计值
        # tp_copy = tp
        tp_copy = sorted(self.tp, key=itemgetter(0), reverse=True)
        cumsum = []
        cumPre = 0
        tp_th = 0
        tp_th_num = 0
        for index, pair in enumerate(tp_copy):
            cumPre += (tp_copy[index][1])
            cumsum.append(cumPre)
            if tp_copy[index][0] > self.threshold:
                tp_th_num += 1
                if tp_copy[index][1] == 1:
                    tp_th += 1
        tp_precision = float(tp_th) / float(tp_th_num)
        return cumsum, tp_th, tp_precision

    def computeAp(self, label):
        num = len(self.tp)
        prec = []  # 精度
        rec = []  # 召回率
        fpr = []  # 负正类率(false postive rate FPR)特异度  ROC曲线使用
        ap = 0  # 平均精度
        if num == 0 or self.all_num_pos == 0:
            return prec, rec, ap
        tp_cumsum, tp_th, tp_precision = self.CumSum_tp()
        fp_cumsum, fp_th, fppw = self.CumSum()
        # Compute precision. Compute recall.
        for i in range(num):
            prec.append(float(tp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))  # 分段精度
            rec.append(float(tp_cumsum[i]) / float(self.all_num_pos))   # 分段recall值
            fpr.append(float(fp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))   # 负正类率(false postive rate FPR)特异度

        if self.roc:
            # 画roc曲线图
            plt.figure('Draw_roc')
            plt.plot(fpr, rec)  # plot绘制折线图
            plt.grid(True)
            plt.xlabel('false positive')
            plt.ylabel('recall')
            plt.draw()  # 显示绘图
            # plt.pause(5)  # 显示5秒
            plt.savefig("class_{}_roc.jpg".format(label))  # 保存图象
            plt.close()

        if self.pr:
            # 画roc曲线图
            plt.figure('Draw_pr')
            plt.plot(rec, prec)  # plot绘制折线图
            plt.grid(True)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.draw()  # 显示绘图
            # plt.pause(5)  # 显示5秒
            plt.savefig("class_{}_pr.jpg".format(label))  # 保存图象
            plt.close()

        fppi = 0
        if self.FPPIW:
            fppi = float(fp_th) / float(self.all_num_pos)

        tmp = 0
        # 11点法求AP
        max_precs = [0.] * 11
        start_idx = num - 1
        j = 10
        while j >= 0:
            i = start_idx
            while i >= 0:
                tmp = j / 10.0
                if rec[i] < tmp:
                    start_idx = i
                    if j > 0:
                        max_precs[j - 1] = max_precs[j]
                        break
                else:
                    if max_precs[j] < prec[i]:
                        max_precs[j] = prec[i]
                i -= 1
            j -= 1
        for iji in range(11):
            ap += max_precs[iji] / 11.0

        # 计算 recall 和 precision
        recall = float(tp_th) / float(self.all_num_pos)
        precision = tp_precision
        return precision, recall, fppi, fppw, ap

    def run(self):
        [prediction_path, gt_path, predictions, groundtruths, file_format]= self.load_all_files()
        aps = 0
        for label in range(1, self.cls):
            self.get_tp_fp(gt_path, prediction_path, groundtruths, predictions, label, file_format)
            precision, recall, fppi, fppw, ap = self.computeAp(label)

            print ("class ", label, " Ap: ", ap)

            if self.precision:
                print("class ", label, "precision: ", precision)
            if self.recall:
                print("class ", label, "recall: ", recall)
            if self.FPPIW:
                print('FPPW: ', fppw, 'FPPI', fppi)
            aps += ap
        mAp = aps / (self.cls - 1)  # mAP计算,考虑背景
        print ("mAp: ", mAp)
        return 0
