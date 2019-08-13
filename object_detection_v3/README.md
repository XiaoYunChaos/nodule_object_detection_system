# nodule_object_detection_system
This is a pulmonary nodule detection system based on a deep learning neural network developed using the Flask framework.

����������ķν�ڼ��ͷ���ϵͳ ��װ˵����
��1��requirements.txt�ļ��������˵�ǰ���������а������Եİ汾���б�
��װ���뻷��������
pip install -r requirements.txt

��2��Flask��Ŀ���У�
  ����Ҫ��python3 ,flask��matplotlib��
  python app.py�������ط���
  ������������ʣ�
  127.0.0.1:5000/   �û�ϵͳ
  127.0.0.1:5000/admin   ��̨����Աϵͳ

 ��3�������ɶ������е�py
 mhd_transform_jpg.py mhd�ļ�ת��

 ��4��127.0.0.1:5000/admin   ��̨����Աϵͳʹ��
  1.�û�����admin  ���룺admin123 ��¼
  2.����Ƿ��û� ��ɾ��
 ��5��127.0.0.1:5000/   �û�ϵͳʹ��
 http://127.0.0.1:5000/   �û���¼    Ĭ���û���root  ���룺root
 http://127.0.0.1:5000/register  �û�ע��
 http://127.0.0.1:5000/index/  �û���ҳ
 http://127.0.0.1:5000/mhd/   mhd�ļ��ϴ�


����������ķν�ڼ��ͷ���ϵͳ  flask��Ŀ˵����
templatesĿ¼���ļ�˵����
��1��admin.htmlΪ��̨����Աϵͳ���
��2��user.htmlΪ����Աϵͳ���
��3��login1.htmlΪ�û���¼�������
��4��register1.htmlΪ�û�ע��������
��5��index.htmlΪ�û�ϵͳ��ҳ���
��6��predict.htmlΪ���ѧϰģ�͵������
��7��set.htmlΪ�û�ͼƬ�������
��8��final.htmlΪͼƬ����ϸ�����
��9��upload_mhd.htmlΪmhd�ļ��ϴ����
staticĿ¼���ļ�˵����
��1��css��js��LoginSpecial��images��img��page1_0~4ͼƬ Ϊǰ�˽�����ʽ�������ʽ���
��2��mhd�ļ��ϴ�Ŀ¼��static/data
    ���������·����static/data/object_detection_img
��3���û���¼�����ͼƬ������static/img(Ϊ�˱����ļ���Ψһ�ԣ��ļ���̨����ʱ���������)
��4��predictImgΪ�������ѧϰģ��Ԥ��󱣴���
��5��tmp_ImgΪ�����û�ͼƬ���ú�Ľ��
userĿ¼˵����
��1��userģ�����ݿ����û����ݴ洢��ϵͳ���ò����û�root��test��
��2��adminģ���̨����Ա���ݴ洢������ԱĬ���û�����admin�����룺admin123��
utilsĿ¼˵�������ѧϰ������������һЩģ������
modelĿ¼˵�������ѧϰ������������һЩģ������

Flask˵����
��1��app.py��flask��Ŀ·�ɽ������߼����
��2��mhd_transform_jpg.py: mhdת�����ԣ����Ե������У�
��3��pascal_label_map.pbtxt��deepLearning ģ����Ҫ
��4��utils.py��������ƹ������Զ����utils�أ������洢���������

