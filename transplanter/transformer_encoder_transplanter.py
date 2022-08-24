from .transplanter_core import copy_weights

class Pair:
    def __init__(self, s_group, t_group=[]):
        self.t_group = t_group
        self.s_group = s_group

    def __str__(self):
        return "Student:\n" + str(self.s_group) + "\nTeacher:\n" + str(self.t_group)

def _group_params(params):
    def _get_group_id(name):
        if name.split(".")[2].isnumeric():
            return name.split(".")[2]
        return name.split(".")[1]
        
    grouped_param_names, current_group = [], []
    current_id = None
    for param_name in params:
        if len(param_name.split(".")) <= 2:
            continue
        block_id = _get_group_id(param_name)
        if block_id == current_id:
            current_group.append(param_name)
        else:
            if len(current_group) > 0:
                grouped_param_names.append(current_group)
            current_group = [param_name]
        current_id = block_id
    grouped_param_names.append(current_group)
    return grouped_param_names


def _map_groups(t_groups : list,
               s_groups : list):
    pairs = []
    pairs.append(Pair(s_group=s_groups[0], t_group=t_groups[0]))
    pairs.append(Pair(s_group=s_groups[1], t_group=t_groups[1]))
    pairs.append(Pair(s_group=s_groups[-1], t_group=t_groups[-1]))
    t_modules, s_modules = t_groups[3:-1], s_groups[3:-1]
    s_ids_matched = []
    for t in range(len(t_modules)):
        s = int(t*len(s_modules)/len(t_modules))
        while s in s_ids_matched:
            s += 1
        pairs.append(Pair(s_group=s_modules[s], t_group=t_modules[t]))
        s_ids_matched.append(s)
    for s in range(len(s_modules)):
        if s in s_ids_matched:
            continue
        pairs.append(Pair(s_group=s_modules[s], t_group=[]))
    return pairs

def _new_encoder_layer(state_dict : dict):
    # TODO(Oleguer): Smart layer initialization
    return state_dict

def _transfer_group(teacher_state_dict : dict,
                    student_state_dict : dict):
    cpt = ".".join(list(teacher_state_dict.keys())[0].split(".")[0:3])  # common prefix teacher
    cps = ".".join(list(student_state_dict.keys())[0].split(".")[0:3])  # common prefix student

    for t_param_name in teacher_state_dict:
        if "loss.weight" in t_param_name:
            continue
        s_param_name = t_param_name.replace(cpt, cps)
        updated_param = None
        if "norm" in t_param_name and ".weight" in t_param_name:
            updated_param = copy_weights(teacher_weights=teacher_state_dict[t_param_name],
                                         student_weights=student_state_dict[s_param_name],
                                         base_bias=1.0)
        else:
            updated_param = copy_weights(teacher_weights=teacher_state_dict[t_param_name],
                                         student_weights=student_state_dict[s_param_name])
        student_state_dict[s_param_name] = updated_param
    return student_state_dict

def transfer_encoder(teacher_state_dict,
                     student_state_dict):
    t_groups = _group_params(teacher_state_dict)
    s_groups = _group_params(student_state_dict)
    mappings = _map_groups(t_groups=t_groups,
                          s_groups=s_groups)
    for indx, mapping in enumerate(mappings):
        teacher_group_dict = {name : teacher_state_dict[name] for name in mapping.t_group}
        student_group_dict = {name : student_state_dict[name] for name in mapping.s_group}
        if mapping.t_group != []:
            student_group_dict = _transfer_group(teacher_group_dict, student_group_dict)
            student_state_dict.update(student_group_dict)
    return student_state_dict


