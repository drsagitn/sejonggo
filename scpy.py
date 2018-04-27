import os
import pexpect
from conf import conf


class DirectoriesSync(object):
    def __init__(self, local, remote):
        self.localDir = os.path.abspath(local)
        self.remoteSite = remote

    def push_remote_site(self):
        r = '-r'
        self.scp(r + ' ' + self.localDir + ' ' + self.remoteSite['user'] + '@' + self.remoteSite['host'] + ':' + self.remoteSite['dest'], self.remoteSite['creds'])

    def pull_remote_site(self):
        r = '-r'
        self.scp(r + ' ' + self.remoteSite['user'] + '@' + self.remoteSite['host'] + ':' + self.remoteSite['dest']+ ' ' + self.localDir, self.remoteSite['creds'])

    def scp(self, cmd, password=None):
        print('scp ' + cmd)
        child = pexpect.spawn('scp ' + cmd, timeout=300)

        expect = ['The authenticity of host', 'Permission denied', 'password', 'refused', pexpect.EOF, pexpect.TIMEOUT]
        index = child.expect(expect)
        if index == 0:
            child.sendline('yes')
        elif index == 1:
            child.sendcontrol('c')
        elif index == 2:
            child.sendline(password)
        elif index == 3:
            print('Connection refused...')
        elif index == 4:
            print('Unexpected end of line')
        elif index == 5:
            print('Timed out...')
            raise Exception
        index = child.expect(['denied', pexpect.EOF])
        if index == 0:
            child.sendcontrol('c')
            print('Permission denied')
            print('Command failed:', 'scp ' + cmd)
        else:
            print(child.before)


def sync_model(model_name=conf['BEST_MODEL']):
    #  I am Training Server or SPE, I want to send a model to ALL Self-play Servers
    local_model_file = os.path.join(conf['MODEL_DIR'], model_name)
    for rs in conf['SELF_PLAY_SERVER']:
        rs_copy = {**rs}  # Shallow copy conf to variable
        rs_copy['dest'] = os.path.join(rs_copy['dest'], conf['MODEL_DIR'])
        print('Syncing model %s to remote place %s' % (local_model_file, rs_copy['host'], rs_copy['dest']))
        d = DirectoriesSync(local_model_file, rs_copy)
        d.push_remote_site()


def retrieve_model(model_name=conf['BEST_MODEL']):
    #  I am Self-play Server, I want to get a model from Training Server
    remote_server = {**conf['TRAINING_SERVER']}  # Shallow copy conf to variable
    remote_server['dest'] = os.path.join(remote_server['dest'], conf['MODEL_DIR'], model_name)
    local_dir = conf['MODEL_DIR']
    print('Retrieving model {} from remote training server {}'.format(model_name, remote_server['host'] + remote_server['dest']))
    d = DirectoriesSync(local_dir, remote_server)
    d.pull_remote_site()


def sync_game_data(operating_dir, model_name, game_name):
    #  I am Self-play Server I want to send my one-game data to Training Server
    remote_server = {**conf['TRAINING_SERVER']}  # Shallow copy conf to variable
    remote_server['dest'] = os.path.join(remote_server['dest'], operating_dir, model_name)
    local_dir = os.path.join(operating_dir, model_name, game_name)
    print('Sending game data to training server. Game ', game_name)
    d = DirectoriesSync(local_dir, remote_server)
    d.push_remote_site()
    print("Finish sending data.")

def sync_eval_game_data(model_name, game_name):
    #  I am SPE Server I want to send my evaluation game data to Training Server
    remote_server = {**conf['TRAINING_SERVER']}  # Shallow copy conf to variable
    remote_server['dest'] = os.path.join(remote_server['dest'], conf['SELF_PLAY_DIR'], model_name)

    local_dir = os.path.join(conf['EVAL_DIR'], model_name, game_name)
    print('Sending eval game data to training server. Game ', game_name)
    d = DirectoriesSync(local_dir, remote_server)
    d.push_remote_site()
    print("Finish sending data.")


def sync_all_game_data(operating_dir, model_name=None):
    #  I am Self-play Server, I want to send all my game data to Training Server
    model_name_list = []
    if model_name is None:
        for model_dir in os.listdir(operating_dir):
            model_name_list.append(model_dir)
    else:
        model_name_list = [model_name]
    remote_server = {**conf['TRAINING_SERVER']}  # Shallow copy conf to variable
    remote_server['dest'] = os.path.join(remote_server['dest'], operating_dir)

    for name in model_name_list:
        local_dir = os.path.join(operating_dir, name)
        if len(os.listdir(local_dir)) > 0: # if dir not empty
            print('Sending game data to training server. Model data ', name)
            d = DirectoriesSync(local_dir, remote_server)
            d.push_remote_site()
    print("Finish sending data.")


def main():
    locals_dir = 'sp_models/best_model.h5'
    remotes = {
        'user': 'miruware',
        'host': '211.180.114.9',
        'dest': '/home/miruware/temp',
        'creds': 'miruware@'
    }
    d = DirectoriesSync(locals_dir, remotes)
    d.push_remote_site()

if __name__ == "__main__":
    sync_model("model_187.h5")
    # sync_all_game_data(conf['SELF_PLAY_DIR'], "model_111")
    # main()
