


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state: # anomaly_state == false
            anomaly_state = True
            for j in range(i, -1, -1): # 向前扫描
                if gt[j] == 0: # 遇到正常点则扫描结束
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)): # 像后扫描
                if gt[j] == 0: # 遇到正常点,扫描结束
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred
