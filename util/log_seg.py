#导入logging模块
import logging

def logger(file_log='test.log', mode='w', fh_print= True):
    '''
    打日志
    :param file_log: 日志文件名，类型string；
    '''
    # 创建一个loggger，并设置日志级别
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 创建一个handler，用于写入日志文件，并设置日志级别，mode:a是追加写模式，w是覆盖写模式
    fh = logging.FileHandler(filename=file_log, encoding='utf-8', mode='w')
    fh.setLevel(logging.INFO)
    
    # 创建一个handler，用于将日志输出到控制台，并设置日志级别
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 定义handler的输出格式
    formatter = logging.Formatter(
        '%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]''-%(levelname)s-[日志信息]: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    if fh_print== True:
        logger.addHandler(ch)
    else:
        pass

    return logger

if __name__=='__main__':
	test_path = 'test'
	logger = logger('{}.log'.format(test_path))
	logger.info("----------------日志输出----------------\n")
