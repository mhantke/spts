import os, numpy, ConfigParser

import logging
logger = logging.getLogger(__name__)

from log import log_and_raise_error,log_warning,log_info,log_debug

def read_configfile(configfile):
    """
    Read configuration file to dictionary
    """
    config = ConfigParser.ConfigParser()
    with open(configfile,"r") as f:
        config.readfp(f)
        confDict = {}
        for section in config.sections(): 
            confDict[section] = {}
            c = config.items(section)
            for (key,value) in c:
                confDict[section][key] = _estimate_class(value)
    return confDict

def write_configfile(configdict, filename):
    """
    Write configuration file from a dictionary
    """
    ls = ["# Configuration file\n# Automatically written by Configuration instance\n\n"]
    for section_name,section in configdict.items():
        if isinstance(section,dict):
            ls.append("[%s]\n" % section_name)
            for variable_name,variable in section.items():
                if (hasattr(variable, '__len__') and (not isinstance(variable, str))) or isinstance(variable, list):
                    ls.append("%s=%s\n" % (variable_name,_list_to_str(variable)))
                else:
                    ls.append("%s=%s\n" % (variable_name,str(variable)))
            ls.append("\n")
    with open(filename, "w") as f:
        f.writelines(ls)

def read_configdict(configdict):
    C = {}
    for k,v in configdict.items():
        if isinstance(v, dict):
            v_new = read_configdict(v)
        else:
            v_new = _estimate_class(v)
        C[k] = v_new
    return C

def _estimate_class(var):
    v = _estimate_type(var)
    if isinstance(v,str):
        v = v.replace(" ","")
        if v.startswith("[") and v.endswith("]"):
            v = _str_to_list(v)
            for i in range(len(v)):
                v[i] = os.path.expandvars(v[i]) if isinstance(v[i], str) else v[i] 
        elif v.startswith("{") and v.endswith("}"):
            v = v[1:-1].split(",")
            v = [w for w in v if len(w) > 0]
            d = {}
            for w in v:
                key,value = w.split(":")
                value = _estimate_type(value)
                if value.startswith("$"):
                    value = os.path.expandvars(value)
                d[key] = value
            v = d
        else:
            if v.startswith("$"):
                v = os.path.expandvars(v)
    return v
        
def _estimate_type(var):
    if not isinstance(var, str):
        return var
    #first test bools
    if var.lower() == 'true':
        return True
    elif var.lower() == 'false':
        return False
    elif var.lower() == 'none':
        return None
    else:
        #int
        try:
            return int(var)
        except ValueError:
            pass
        #float
        try:
            return float(var)
        except ValueError:
            pass
        #string
        try:
            return str(var)
        except ValueError:
            raise NameError('Something messed up autocasting var %s (%s)' % (var, type(var)))

def _str_to_list(s):
    if s.startswith("[") and s.endswith("]"):
        if s[1:-1].startswith("[") and s[1:-1].endswith("]"):
            return _str_to_list(s[1:-1])
        else:
            l = s[1:-1].split(",")
            l = [_estimate_type(w) for w in l if len(w) > 0]
            return l
    else:
        return s
       
def _list_to_str(L):
    if (hasattr(L, '__len__') and (not isinstance(L, str))) or isinstance(L, list):
        s = ""
        for l in L:
            s += _list_to_str(l)
            s += ","
        s = "[" + s[:-1] + "]"
        return s
    else:
        return str(L)
