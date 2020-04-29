def calc_iou(boxProposal1, boxProposal2):

    assert boxProposal1['x1'] < boxProposal1['x2']
    assert boxProposal1['y1'] < boxProposal1['y2']
    assert boxProposal2['x1'] < boxProposal2['x2']
    assert boxProposal2['y1'] < boxProposal2['y2']
    # the coordinate of the intersection box
    x_left = max(boxProposal1['x1'], boxProposal2['x1'])
    y_top = max(boxProposal1['y1'], boxProposal2['y1'])
    x_right = min(boxProposal1['x2'], boxProposal2['x2'])
    y_bottom = min(boxProposal1['y2'], boxProposal2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    boxProposal1_area = (boxProposal1['x2'] - boxProposal1['x1']) * (boxProposal1['y2'] - boxProposal1['y1'])
    boxProposal2_area = (boxProposal2['x2'] - boxProposal2['x1']) * (boxProposal2['y2'] - boxProposal2['y1'])
    iou = intersect_area / float(boxProposal1_area + boxProposal2_area - intersect_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou