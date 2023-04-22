import logging
import os
from datetime import datetime

import numpy as np
from django.utils import timezone
import json

from ring.forms.pulse_data import FirstTrainData, SecondTrainData, Token, PredictData, TrainData, TrainImOrTime, \
    SchedulerUpdate, EmoLabelUpload, NewPredictData
from ring.models import CalibrationData
from ring.models.health_history import MinHealth
from ring.models.management_movies import TrainDataMoviesPlan, TrainDataMoviesData
from ring.models.ppg_emotion_map import PpgEmotionMap
from ring.models.study_status import StudyStatus
from ring.utils.data_util import get_emo, get_emo_bp_20s, get_study_status, get_user_map, save_emo, new_get_emo
from ring.utils.mongodb_util import hour_scheduler_update, day_scheduler_update, month_scheduler_update
from ..utils.response import corr_response, err_response
from ..utils.decorator import validate_forms, validate_token
from ..algorithm import blood_pressure_ysl_v1, ppg_emotion_100hz
from ..algorithm.ppg_bp_ import blood_pressure_kpts

logger = logging.getLogger('root')


@validate_forms(FirstTrainData)
@validate_token()
def first_train_data_upload(request, data):
    """
    上传第一次标定数据
    :param request:
    :param data:
        | 参数         | 必选 | 类型     | 说明                   |
        | ------------ | ---- | -------- | ---------------------- |
        | token        | true | string   | 令牌                   |
        | start_time   | true | Datetime | 采集开始时间           |
        | end_time     | true | Datetime | 采集结束时间           |
        | data         | true | list     | 脉搏波数据             |
        | device_code  | true | string   | 设备码，长度17，0-9A-F |
        | video_id     | true | string   | 视频id                 |
        | emo_category | true | integer  | 情绪类型(0-6)          |
        | emo_level    | true | integer  | 情绪强度(1-5)          |
        | is_cover     | true | boolean  | 是否覆盖上传,1覆盖     |

        - is_cover是为了保证第一次标定数据有效且唯一(方便第一次建模) 如果用户重复观看视频就需要将它设为1，覆盖之前的数据
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    # todo:数据校验，结束时间必须比开始时间大三分钟以上，
    #  用户存在性校验，得根据mysql的用户表校验，包括设备号
    #  也是，然后视频分类也得根据mysql记录查询
    user = data.pop('user')
    is_cover = data.pop('is_cover')

    calibration_data = CalibrationData.objects.filter(
        user_id=user.user_id, similarity=0, emo_category=data['emo_category'])

    # 保证基本模型每个情绪有且仅有一条
    if is_cover and calibration_data.count() > 0:
        calibration_data.delete()
    if (not is_cover) and calibration_data.count() > 0:
        return err_response('1015', ' 绪基本模型数据已存在')

# 训练数据过来先进行特征提取并保存结果
start_time = timezone.now()
    try:
        feature, emo1 = ppg_emotion_100hz.feature_generator(data['data'], 0)
    except Exception:
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    if len(feature) == 0:
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    finish_time = timezone.now()
    feature_exectime = (finish_time - start_time).total_seconds()
    logger.info('user: {} emo: {} length: {} feature generator time: {}'.format(
        user.user_id, data['emo_category'], len(data['data']), feature_exectime))

    data['feature'] = feature  # str(feature)
    data['data'] = str(data['data'])
    data['user_id'] = user.user_id
    data['is_used'] = 0
    data['similarity'] = 0  # 基本模型的数据都设为0，便于管理以及区分二次标定的数据
    try:
        CalibrationData.objects.create(**data)
    except Exception:
        return err_response('1103', 'mongo保存数据失败')

    return corr_response()


@validate_forms(FirstTrainData)
@validate_token()
def first_new_train_data_upload(request, data):
    """
    上传第一次标定数据
    :param request:
    :param data:
        | 参数         | 必选 | 类型     | 说明                   |
        | ------------ | ---- | -------- | ---------------------- |
        | token        | true | string   | 令牌                   |
        | start_time   | true | Datetime | 采集开始时间           |
        | end_time     | true | Datetime | 采集结束时间           |
        | data         | true | list     | fm数据             |
        | device_code  | true | string   | 设备码，长度17，0-9A-F |
        | video_id     | true | string   | 视频id                 |
        | emo_category | true | integer  | 情绪类型(0-6)          |
        | emo_level    | true | integer  | 情绪强度(1-5)          |
        | is_cover     | true | boolean  | 是否覆盖上传,1覆盖     |

        - is_cover是为了保证第一次标定数据有效且唯一(方便第一次建模) 如果用户重复观看视频就需要将它设为1，覆盖之前的数据
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    # todo:数据校验，结束时间必须比开始时间大三分钟以上，
    #  用户存在性校验，得根据mysql的用户表校验，包括设备号
    #  也是，然后视频分类也得根据mysql记录查询
    logger.info('data: {} '.format(data))
    user = data.pop('user')
    is_cover = data.pop('is_cover')
    logger.info('user: {} \n is_cover: {}  \n'.format(user.user_id, is_cover))
    calibration_data = CalibrationData.objects.filter(user_id=user.user_id, similarity=0, emo_category=data['emo_category'])
    logger.info('calibration_data: {} \n'.format(calibration_data))
# 保证基本模型每个情绪有且仅有一条
if is_cover and calibration_data.count() > 0:
        calibration_data.delete()
    if (not is_cover) and calibration_data.count() > 0:
        return err_response('1015', ' 绪基本模型数据已存在')

    # 训练数据过来先进行特征提取并保存结果
    start_time = timezone.now()
    try:
        Fmext = ppg_emotion_100hz.temperal_feature_extender(np.array(data['data']))
    except Exception:
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    if len(Fmext) == 0:
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    emo_category = data['emo_category']
    feature, emo1 = ppg_emotion_100hz.feature_transformer_frequential(Fmext, emo_category)
    finish_time = timezone.now()
    feature_exectime = (finish_time - start_time).total_seconds()
    logger.info('user: {}   emo: {} length: {} feature generator time: {}'.format(
        user.user_id, data['emo_category'], len(feature), feature_exectime))
    data['feature'] = feature
    data['data'] = str(data['data'])
    data['user_id'] = user.user_id
    data['is_used'] = 0
data['similarity'] = 0   # 基本模型的数据都设为0，便于管理以及区分二次标定的数据
logger.info('data: {} '.format(data))
    try:
        CalibrationData.objects.create(**data)
    except Exception:
        return err_response('1103', 'mongo保存数据失败')
    return corr_response()


@validate_forms(SecondTrainData)
@validate_token()
def second_train_data_upload(request, data):
    """
    上传第二次标定数据,这里会对该情绪数据和第一次上传的其他情绪数据重新建模
    然后校验之前上传的该情绪的所有数据，找出两组具有一定相似性的数据当作最终
    模型的标准数据，可多次上传
    :param request:
    :param data:
        | 参数         | 必选 | 类型     | 说明                 |
        | ------------ | ---- | -------- | -------------------- |
        | token        | true | string   | 令牌                 |
        | start_time   | true | Datetime | 采集开始时间         |
        | end_time     | true | Datetime | 采集结束时间         |
        | data         | true | list     | 脉搏波数据           |
        | device_code  | true | string   | 设备码长度17，0-9A-F |
        | video_id     | true | string   | 视频id               |
        | emo_category | true | integer  | 情绪类型(0-6)        |
        | emo_level    | true | integer  | 情绪强度(1-5)        |
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    start_time = timezone.now()
    try:
        feature, emo1 = ppg_emotion_100hz.feature_generator(data['data'], 0)
    except Exception:
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    if len(feature) == 0:
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    finish_time = timezone.now()
    feature_exectime = (finish_time - start_time).total_seconds()
    logger.info('user: %s emo: %s length: %s feature generator time: %s'
                % (user.user_id, data['emo_category'], len(data['data']), feature_exectime))

# 直接读模型校验看看
# pca, lda, clf = ppg_emotion_100hz.load_classifer('./models/18729396280')
    # print(ppg_emotion_100hz.is_emotion_correct(feature, pca, lda, clf, data['emo_category']))

   # 2.找出基本模型的其他情绪特征值
    other_emo = CalibrationData.objects.filter(
        user_id=user.user_id, is_used=True, similarity=0, emo_category__ne=data['emo_category']
    ).only('feature', 'emo_category')

    FmFre = feature
    FmEmo = [data['emo_category'] for i in range(len(feature))]
    for emo in other_emo:
        FmFre.extend(emo.feature)
        FmEmo.extend([emo.emo_category for i in range(len(emo.feature))])

    # 3.建模
    pca, lda, clf = ppg_emotion_100hz.emotion_classifier_trainer(FmFre, FmEmo)

    # 4.找出该情绪历史特征数据
    history_data = CalibrationData.objects.filter(user_id=user.user_id, emo_category=data['emo_category']) \
        .only('feature', 'emo_category', 'similarity', 'is_used')

    # 5.校验
    data['feature'] = feature
    data['data'] = str(data['data'])
    data['user_id'] = user.user_id
    data['is_used'] = 0
    for emo_data in history_data:
        if ppg_emotion_100hz.is_emotion_correct(emo_data.feature, pca, lda, clf, data['emo_category']):
            # 如果校验成功，更改这两条数据状态similarity=0,这样就得到了一组具有相似性的数据
            model_data = history_data.filter(similarity=0).first()
            data['similarity'] = 0
            # 将匹配的历史数据交换基础模型的数据,并保存最新提交的数据
            try:
                if emo_data.similarity != 0:
                    emo_data.similarity, model_data.similarity = model_data.similarity, emo_data.similarity
                    emo_data.is_used, model_data.is_used = model_data.is_used, emo_data.is_used
                    model_data.save()
                    emo_data.save()
                CalibrationData.objects.create(**data)
            except Exception:
                return err_response('1103', 'mongo保存数据失败')
            return corr_response()

    # 6.校验未通过，将情绪状态设为该情绪历史标定数据条数之和
    data['similarity'] = len(history_data)
    CalibrationData.objects.create(**data)
    return err_response('1017', '校验未通过，请继续观看该情绪其他视频')

@validate_forms(SecondTrainData)
@validate_token()
def second_new_train_data_upload(request, data):
     """
    上传第二次标定数据,这里会对该情绪数据和第一次上传的其他情绪数据重新建模
    然后校验之前上传的该情绪的所有数据，找出两组具有一定相似性的数据当作最终
    模型的标准数据，可多次上传
    :param request:
    :param data:
        | 参数         | 必选 | 类型     | 说明                 |
        | ------------ | ---- | -------- | -------------------- |
        | token        | true | string   | 令牌                 |
        | start_time   | true | Datetime | 采集开始时间         |
        | end_time     | true | Datetime | 采集结束时间         |
        | data         | true | list     | fm数据           |
        | device_code  | true | string   | 设备码长度17，0-9A-F |
        | video_id     | true | string   | 视频id               |
        | emo_category | true | integer  | 情绪类型(0-6)        |
        | emo_level    | true | integer  | 情绪强度(1-5)        |
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    user = data.pop('user')
    user_id = user.user_id
    logger.info(' user_id: {} \n'.format(user_id))
    if user.model_times != 1:
        return err_response('1016', '请先建立基础模型再进行第二次标定')
# 1.提取特征
    start_time = timezone.now()
    try:
        Fmext= ppg_emotion_100hz.temperal_feature_extender(np.array(data['data']))
        logger.info('Fmext: {} \n'.format(Fmext))
        #feature, emo1 = ppg_emotion_100hz.feature_generator(data['data'], 0)
    except Exception:
        logger.info('Exception: {} \n'.format(Exception))
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    if len(Fmext) == 0:
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    emo_category = data['emo_category']
    logger.info('emo_category: {} \n'.format(emo_category))
    try:
       feature, emo1 = ppg_emotion_100hz.feature_transformer_frequential(Fmext, emo_category)
       logger.info('feature: {} \n'.format(feature))
    except Exception:
       logger.info('Exception: {} \n'.format(Exception))
       return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    finish_time = timezone.now()
    feature_exectime = (finish_time - start_time).total_seconds()
    logger.info('user: %s emo: %s length: %s feature generator time: %s'
                % (user.user_id, data['emo_category'], len(feature), feature_exectime))

# 直接读模型校验看看
model_path = './models/' + user_id
    logger.info('model_path : {} \n'.format(model_path))
    try:
        pca, lda, clf = ppg_emotion_100hz.load_classifer(model_path)
        logger.info('pca: {} \n'.format(pca))
    except Exception:
        logger.info('Exception: {} \n'.format(Exception))
        return err_response('1014', '璇绘ā鍨嬪け璐?)
    try:
        flag = ppg_emotion_100hz.is_emotion_correct(feature, pca, lda, clf, emo_category)
        logger.info('flag : {} \n'.format(flag))
    except Exception:
        logger.info('Exception: {} \n'.format(Exception))
        return err_response('1014', 'correct澶辫触')
    if flag:
        return corr_response()

    # 2.找出基本模型的其他情绪特征值
    other_emo = CalibrationData.objects.filter(
        user_id=user.user_id, is_used=True, similarity=0, emo_category__ne=data['emo_category']
    ).only('feature', 'emo_category')

    FmFre = feature
    FmEmo = [emo_category for i in range(len(feature))]
    for emo in other_emo:
        FmFre.extend(emo.feature)
        FmEmo.extend([emo.emo_category for i in range(len(emo.feature))])
    logger.info('2.鎵惧嚭鍩烘湰妯″瀷鐨勫叾浠栨儏缁壒寰佸€?\n')
    # 3.建模
    pca, lda, clf = ppg_emotion_100hz.emotion_classifier_trainer(FmFre, FmEmo)
    logger.info('3.寤烘ā \n')

    # 4.找出该情绪历史特征数据
    history_data = CalibrationData.objects.filter(user_id=user.user_id, emo_category=data['emo_category']) \
        .only('feature', 'emo_category', 'similarity', 'is_used')
    logger.info('4.鎵惧嚭璇ユ儏缁巻鍙茬壒寰佹暟鎹?\n')
    # 5.校验
    data['feature'] = feature
    data['data'] = str(data['data'])
    data['user_id'] = user.user_id
    data['is_used'] = 0
    for emo_data in history_data:
        if ppg_emotion_100hz.is_emotion_correct(emo_data.feature, pca, lda, clf, data['emo_category']):
             # 如果校验成功，更改这两条数据状态similarity=0,这样就得到了一组具有相似性的数据
            model_data = history_data.filter(similarity=0).first()
            data['similarity'] = 0
             # 将匹配的历史数据交换基础模型的数据,并保存最新提交的数据
            try:
                if emo_data.similarity != 0:
                    emo_data.similarity, model_data.similarity = model_data.similarity, emo_data.similarity
                    emo_data.is_used, model_data.is_used = model_data.is_used, emo_data.is_used
                    model_data.save()
                    emo_data.save()
                CalibrationData.objects.create(**data)
            except Exception:
                return err_response('1103', 'mongo保存数据失败')
            return corr_response()
    logger.info('5.鏍￠獙 \n')
# 6.校验未通过，将情绪状态设为该情绪历史标定数据条数之和
data['similarity'] = len(history_data)
    CalibrationData.objects.create(**data)
    return err_response('1017', '校验未通过，请继续观看该情绪其他视频')

@validate_forms(Token)
@validate_token()
def create_first_model(request, data):
    """
    建立基础模型
    :param request:
    :param data:
        | 参数  | 必选 | 类型   | 说明 |
        | ----- | ---- | ------ | ---- |
        | token | true | string | 令牌 |
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    #logger.info('data:{}\n'.format(data))
    user = data['user']
    user_id = user.user_id
    if user.model_times != 0:
        return err_response('1020', '已存在基础模型，无需重复创建')

    base_model = CalibrationData.objects.filter(user_id=user_id, is_used=False, similarity=0) \
        .only('feature', 'emo_category')  # .all_fields()
    #logger.info('base_model: {} \n'.format(base_model))
    logger.info('base_model length: %s \n' % (len(base_model)))
    if len(base_model) != 7:
        return err_response('1018', '请先采集完七种情绪的有效数据在建立基础模型')
    else:
        FmFre, FmEmo = list(), list()
        # base_model = base_model.only('feature', 'emo_category')
        # start_time = timezone.now()
        for emo in base_model:
            FmFre.extend(emo.feature)
            FmEmo.extend([emo.emo_category for i in range(len(emo.feature))])
        logger.info('FmFre: {} FmEmo:{}'.format(FmFre,FmEmo))
        # end_time = timezone.now()
        # print((end_time - start_time).total_seconds(), len(FmFre))
        try:
            pca, lda, clf = ppg_emotion_100hz.emotion_classifier_trainer(FmFre, FmEmo)
        except Exception as ex:
            logger.info('ppg_emotion_100hz.emotion_classifier_trainer err {}'.format(ex))
        #logger.info('pca:{} \n'.format(pca))
        model_path = './models/' + user_id
        os.makedirs(model_path, exist_ok=True)
        logger.info('makedirs success \n')
        # ppg_emotion_100hz.save_classifer(pca, lda, clf, model_path)
        try:
            ppg_emotion_100hz.save_classifer(pca, lda, clf, model_path)
        except Exception as ex:
            logger.info(' ppg_emotion_100hz.save_classifer err {} \n'.format(ex))
            return err_response('1105', '创建个人基础模型失败，请重新创建')
# 创建成功，修改对应数据状态
base_model.update(is_used=True)
    user.model_times = 1
    user.save()
    return corr_response()


@validate_forms(Token)
@validate_token()
def create_second_model(request, data):
    """
    创建第二次标定模型（完整模型）
    :param request:
    :param data:
        | 参数  | 必选 | 类型   | 说明 |
        | ----- | ---- | ------ | ---- |
        | token | true | string | 令牌 |
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    user = data['user']
    user_id = user.user_id

    if user.model_times != 1:
        return err_response('1016', '请先建立基础模型')
    else:
        model_data = CalibrationData.objects.filter(user_id=user_id, similarity=0) \
            .only('feature', 'emo_category', 'is_used')
        second_data = model_data.filter(is_used=False)
        if second_data.count() < 7:
            return err_response('1019', '提升关尚有未校验成功的情绪')
        else:
            FmFre, FmEmo = list(), list()
            for emo in model_data:
                FmFre.extend(emo.feature)
                FmEmo.extend([emo.emo_category for i in range(len(emo.feature))])
            pca, lda, clf = ppg_emotion_100hz.emotion_classifier_trainer(FmFre, FmEmo)
            model_path = './models/' + user_id
            os.makedirs(model_path, exist_ok=True)
            try:
                ppg_emotion_100hz.save_classifer(pca, lda, clf, model_path)
            except Exception:
                return err_response('1105', '创建个人模型失败，请重新创建')

    user.model_times = 2
    user.save()

    return corr_response()


@validate_forms(Token)
@validate_token()
def get_train_status(request, data):
    """
    获取标定状态
    :param request:
    :param data:
        | 参数  | 必选 | 类型   | 说明 |
        | ----- | ---- | ------ | ---- |
        | token | true | string | 令牌 |
    :return:
        | 返回字段      | 字段类型 | 说明                   |
        | ------------- | -------- | ---------------------- |
        | status        | bool     | 返回结果状态true/false |
        | first_status  | list     | 第一次标定状态         |
        | second_status | list     | 第二次标定状态         |
    """

    user = data['user']
    user_id = user.user_id

    first_status = [0 for i in range(8)]
    second_status = [0 for i in range(8)]
    train_data = CalibrationData.objects.filter(user_id=user_id) \
        .only('emo_category', 'is_used', 'similarity')

    if user.model_times == 2:  # 建模完成
        first_status = [1 for i in range(8)]
        second_status = [1 for i in range(8)]
    elif user.model_times == 1:  # 基本模型
        first_status = [1 for i in range(8)]
        # 基本模型已存在，就找出第二次标定已成功的情绪
        second_data = train_data.filter(is_used=False, similarity=0)
        for second in second_data:
            second_status[second.emo_category] = 1
elif user.model_times == 0:  # 未建模
    # start_time = timezone.now()
        first_train = train_data.filter(similarity=0)
        second_time = timezone.now()
        for first in first_train:
            first_status[first.emo_category] = 1
        # end_time = timezone.now()
        # print((end_time - start_time).total_seconds(), (end_time-second_time).total_seconds())

    return corr_response({'first_status': first_status, 'second_status': second_status})


@validate_forms(PredictData)
@validate_token()
def predict_data_upload(request, data):
    """
    上传预测数据, 返回预测结果
    :param request:
    :param data:
        | 参数        | 必选 | 类型     | 说明                 |
        | ----------- | ---- | -------- | -------------------- |
        | token       | true | string   | 令牌                 |
        | start_time  | true | datetime | 数据采集开始时间     |
        | end_time    | true | datetime | 数据采集结束时间     |
        | data        | true | list     | 数据                 |
        | device_code | true | string   | 设备码长度17，0-9A-F |
        | step_count  | true | integer  | 步数                 |
    :return:
        | 返回字段 | 字段类型 | 说明                                      |
        | -------- | -------- | ----------------------------------------- |
        | status   | bool     | 返回结果状态true/false                    |
        | emo      | str      | 情绪预测结果(还会作修改，先返回str用着把) |
        | hr       | float    | 心率                                      |
        | bp       | str      | 血压                                      |
    """
    user = data.pop('user')
    # 获取情绪结果数据
    flag, fm, emo_result = get_emo(user, data['data'])
    try:
        # 获取心率
        hr_result = ppg_emotion_100hz.heart_rate(np.array(fm))
    except Exception:
        hr_result = 0
    bp_result = blood_pressure_ysl_v1.predict_blood_pressure_ppg(data['data'])

    if bp_result['isSuccessful'] == 1:
        bp_result = bp_result['dbp'] + '|' + bp_result['sbp']
    else:
        bp_result = '0|0'

    try:
        hr_result = float(hr_result)
        data['emo_result'] = ''.join([str(i) for i in emo_result]) if flag else ''
        user_id = user.user_id
        step_count = data.get('step_count', 0)
        health_min_data = {'user_id': user_id, 'start_time': data['start_time'], 'end_time': data['end_time'],
                           'emo_min': [emo_result.count(i) / len(emo_result) for i in range(0, 7)] if flag else [],
                           'bp_min': bp_result, 'hr_min': hr_result, 'step_count': step_count}
        emo = health_min_data['emo_min']
        study_status = get_study_status(emo, hr_result)
        health_min_data['study_status'] = study_status
        MinHealth.objects.create(**health_min_data)
    except Exception:
        return err_response('1103', 'mongo保存数据失败')
    try:
        study_status_data = StudyStatus.objects.get(user_id=user_id)
        status = study_status_data.study_status
        rate = study_status_data.count / (study_status_data.count + 1)
        study_status_data.study_status = status * rate + study_status * (1 - rate)
        study_status_data.count += 1
    except Exception:
        study_status_data = StudyStatus(user_id=user_id, study_status=study_status, count=1)
    try:
        study_status_data.save()
    except Exception:
        return err_response('1103', 'mongo保存数据失败')

    result = {'emo': data['emo_result'], 'hr': hr_result, 'bp': bp_result, 'study_status': study_status}
    return corr_response(result)


@validate_forms(TrainData)
@validate_token()
def train_data_upload(request, data):
    """
    实验数据上传
    :param request:
    :param data:
        | 参数       | 必选 | 类型     | 说明                    |
        | ---------- | ---- | -------- | ----------------------- |
        | token      | true | string   | 令牌                    |
        | plan_id    | true | string   | 电影计划id              |
        | start_time | true | datetime | 数据采集开始时间        |
        | end_time   | true | datetime | 数据采集结束时间        |
        | data       | true | list     | 数据                    |
        | label      | true | string   | 标签                    |
        | end        | true | bool     | ture:结束,false：上传中 |
    :return:
        | 返回字段 | 字段类型 | 说明                                     |
        | -------- | -------- | ---------------------------------------- |
        | status   | bool     | 返回结果状态true/false                   |
        | start    | bool     | true：实验已开始，false：实验结束/未开始 |
    """
    movie = TrainDataMoviesPlan.objects.get(plan_id=data['plan_id'])
    time_now = datetime.now()
    user = data.pop('user')
    # 建立用户目录，phone
    user_path = './train/' + user.phone
    if not os.path.exists(user_path): # 如果路径不存在
        os.makedirs(user_path)
    # 建立用当前日期目录，datetime
    time_path = user_path + '/' + time_now.strftime('%Y-%m-%d')
    if not os.path.exists(time_path):
        os.makedirs(time_path)

    if data['end']:
        frequential_path = time_path + '/frequential'
        emos = TrainDataMoviesData.objects.filter(phone=user.phone, plan_id=data['plan_id']).only('emo')
        feature = ''.join([''.join([str(j) for j in i.emo]) for i in emos])
        with open(frequential_path, 'a', encoding='utf-8') as f:
            f.write(movie.name + feature + '\n')
    else:
        if time_now < movie.start_time or time_now > movie.end_time:
            return corr_response({'start': False})
        # 建立当前数据目录，phone
        data_path = time_path + '/' + movie.name + '-' + str(data['count'])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if len(data['data']) > 5:
            with open(data_path + '/' + 'row_data', 'w', encoding='utf-8') as f:
                f.write(json.dumps(data['data']))
            try:
                fm = ppg_emotion_100hz.feature_transformer_temperal(ppg_emotion_100hz.feature_extractor(data['data']))
            except Exception:
                fm = ['feature_transformer_temperal error']
            with open(data_path + '/' + 'Fm_result', 'w', encoding='utf-8') as f:
                f.write(json.dumps(fm))
        with open(data_path + '/' + 'label', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data['label']))

        movie_data = TrainDataMoviesData.objects.filter(phone=user.phone, start_time=data['start_time'])
        if movie_data.count():
            movie_data = movie_data[0]
            movie_data.label = data['label']
        else:
            flag, fm, emo_result = get_emo(user, data['data'])
            movie_data = TrainDataMoviesData(
                phone=user.phone, plan_id=data['plan_id'], name=movie.name, start_time=data['start_time'],
                end_time=data['end_time'], count=data['count'], label=data['label'], data=data['data'],
                emo=[int(i) for i in emo_result])
        try:
            movie_data.save()
        except Exception:
            return err_response('1103', 'mongo保存数据失败')

    return corr_response({'start': True})


@validate_forms(TrainImOrTime)
@validate_token()
def train_data_upload_im_or_time(request, data):
    """
    实验上传截图/时间
    :param request:
        | 参数      | 必选  | 类型     | 说明         |
        | --------- | ----- | -------- | ------------ |
        | token     | true  | string   | 令牌         |
        | plan_id   | true  | string   | 电影计划id   |
        | name      | true  | string   | 电影名       |
        | movie_url | true  | string   | 电影链接     |
        | image_url | false | string   | 截图地址     |
        | time      | false | datetime | 电影开始时间 |
    :param data:
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    user = data.pop('user')
    user_path = './train/' + user.phone
    if not os.path.exists(user_path):  # 如果路径不存在
        os.makedirs(user_path)
    # 建立用当前日期目录，datetime
    time_path = user_path + '/' + datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists(time_path):
        os.makedirs(time_path)
    frequential_path = time_path + '/frequential'
    if not os.path.isfile(frequential_path):
        with open(frequential_path, 'w', encoding='utf-8') as f:
            f.write(str(data['time']) + '\n')
    movie_path = time_path + '/movies'
    if not os.path.isfile(movie_path):
        with open(movie_path, 'w', encoding='utf-8') as f:
            f.write(data['name'] + '\n' + data['movie_url'] + '\n' + str(data['image_url']) + '\n' +
                    str(str(data['time']) + '\n'))
    else:
        with open(movie_path, 'a', encoding='utf-8') as f:
            f.write(data['name'] + '\n' + data['movie_url'] + '\n' + str(data['image_url']) + '\n' +
                    str(str(data['time']) + '\n'))

    return corr_response()


@validate_forms(SchedulerUpdate)
@validate_token()
def scheduler_update(request, data):
    """
    定时任务更新
    :param request:
    :param data:
    | 参数           | 必选 | 类型          | 说明                         |
    | -------------- | ---- | ------------- | ---------------------------- |
    | token          | true | string        | 令牌                         |
    | start_time_set | true | set(datetime) | 任务开始时间                 |
    | type           | true | string        | 任务类型（hour，day，month） |
    :return:
    | 返回字段 | 字段类型 | 说明                   |
    | -------- | -------- | ---------------------- |
    | status   | bool     | 返回结果状态true/false |
    """
    scheduler_type = data['type']
    if scheduler_type not in ['hour', 'day', 'month']:
        return err_response('1033', '表单数据验证未通过')
    if scheduler_type == 'hour':
        for start_time in data['start_time_set']:
            hour_scheduler_update(data['user'].user_id, datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
    elif scheduler_type == 'day':
        for start_time in data['start_time_set']:
            day_scheduler_update(data['user'].user_id, datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
    elif scheduler_type == 'month':
        for start_time in data['start_time_set']:
            month_scheduler_update(data['user'].user_id, datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
    return corr_response()


@validate_forms(EmoLabelUpload)
@validate_token()
def emo_label_upload(request, data):
    user = data.pop('user')
    return save_emo(user, data['data'], data['label'])


@validate_forms(NewPredictData)
@validate_token()
def new_predict_data_upload(request, data):
   """
        ##### 参数
    | 参数        | 必选 | 类型     | 说明                 |
    | ----------- | ---- | -------- | -------------------- |
    | token       | true | string   | 令牌                 |
    | start_time  | true | datetime | 数据采集开始时间     |
    | end_time    | true | datetime | 数据采集结束时间     |
    | data        | true | list     | 数据(二维数组)                |
    | device_code | true | string   | 设备码长度17，0-9A-F |
    | step_count  | true | integer  | 步数                 |
    | tp          | true |  float   | 体温                 |
    ##### 返回参数

    | 返回字段     | 字段类型 | 说明                                      |
    | ------------ | -------- | ----------------------------------------- |
    | status       | bool     | 返回结果状态true/false                    |
    | emo          | str    | 情绪预测结果 |
    | hr           | float    | 心率                                      |
    | bp           | str      | 血压                                      |
    | study_status | float    | 学习状态                                  |
    """
    user = data.pop('user')
    #logger.info(' user: {} fm_old: {} length: {} '.format(user.user_id, data['data'], len(data['data'])))
    fm = ppg_emotion_100hz.temperal_feature_extender(np.array(data['data']))
    #logger.info('fm_ext:{}'.format(fm))
    # 获取情绪结果数据
    flag,  emo_result = new_get_emo(user, np.array(fm))
    #logger.info('flag: {} user: {} fm_ext: {} length: {} emo_result: {}'.format(flag, user.user_id, fm, len(fm), emo_result))
    try:
        # 获取心率
        hr_result = ppg_emotion_100hz.heart_rate(np.array(fm))
        #logger.info('fm: {} hr_result: {}'.format(fm, hr_result))
    except Exception:
        hr_result = 0
    try:
        bp_result = blood_pressure_kpts.predict_blood_pressure(np.array(data['data']), user.user_id)
        bp_result = json.loads(bp_result)
        #logger.info('bp_result: {}'.format(bp_result))
        """                                              
        bp_result = blood_pressure_ysl_v1.predict_blood_pressure_ppg(np.array(fm))
        """
        if bp_result['isSuccessful']:
            bp_result = str(int(float(bp_result['dbp']))) + '|' + str(int(float(bp_result['sbp'])))
        else:
            bp_result = '0|0'
    except Exception as ex:
        logger.error('bp_result computer error {}',ex)
        bp_result = '0|0'
    try:
        hr_result = float(hr_result)
        data['emo_result'] = ''.join([str(i) for i in emo_result]) if flag else ''
        user_id = user.user_id
        step_count = data.get('step_count', 0)
        tp = data.get('tp', 0)
        tp = float(tp)
        #logger.info('tp: {}'.format(tp))
        health_min_data = {'user_id': user_id, 'start_time': data['start_time'], 'end_time': data['end_time'],
                           'emo_min': [emo_result.count(i) / len(emo_result) for i in range(0, 7)] if flag else [],
                           'bp_min': bp_result, 'hr_min': hr_result, 'step_count': step_count, 'tp_min': tp}
        emo = health_min_data['emo_min']
        study_status = get_study_status(emo, hr_result)
        health_min_data['study_status'] = study_status
        MinHealth.objects.create(**health_min_data)
    except Exception:
        return err_response('1103', 'mongo保存数据失败')
    try:
        study_status_data = StudyStatus.objects.get(user_id=user_id)
        status = study_status_data.study_status
        rate = study_status_data.count / (study_status_data.count + 1)
        study_status_data.study_status = status * rate + study_status * (1 - rate)
        study_status_data.count += 1
    except Exception:
        study_status_data = StudyStatus(user_id=user_id, study_status=study_status, count=1)
    try:
        study_status_data.save()
    except Exception:
        return err_response('1103', 'mongo保存数据失败')
    result = {'emo': data['emo_result'], 'hr': hr_result, 'bp': bp_result, 'study_status': study_status}
    return corr_response(result)

