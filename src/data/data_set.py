# -*- coding: utf-8 -*-


class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    oneHot : bool
    targetDigit : string
    """

    def __init__(self, data, oneHot=False, targetDigit='7'):

        # The label of the digits is always the first fields
        self.input = 1.0*data[:, 1:]/255.0
        self.label = data[:, 0]
	print('label in dataset' + str(self.label))
        self.oneHot = oneHot
        #self.targetDigit = targetDigit

        # Transform all labels which is not the targetDigit to False,
        # The label of targetDigit will be True,
        if oneHot:
            self.label = list(map(lambda a: 1 if str(a) == targetDigit else 0,
                                  self.label))

    def __iter__(self):
        return self.input.__iter__()
