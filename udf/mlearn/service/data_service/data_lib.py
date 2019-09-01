import hdfs
import pandas as pd
import numpy as np
import os
import shutil
import subprocess
import re
from pyhive import hive
import prestodb

os.environ.update({'JAVA_HOME': '/usr/local/jdk1.8.0_74'})

class DataLib:
    def __init__(self, hive_user='yangkai', hive_pwd='8w8hu2R6QVfXCmB8'):
        self.hive_user = hive_user
        self.hive_pwd  = hive_pwd
        self.hdfs_root = 'hdfs://nameservicestream/user/hive/warehouse'
        self.hive_host = '10.66.245.156'
        self.hive_port = 10000
        self.presto_host = '10.105.110.176'
        self.presto_port = 8080
        self.presto_user = 'mapeng'
        self.presto_catalog = 'hive'
        self.presto_schema  = 'dwb'


    def _execute_by_pyhive(self, sql, query_type='select'):
        """ 通过pyhive执行sql
        :param sql         : 需要执行的sql
        :param query_type  : query的类型
        :return:
        """
        try:
            conn = hive.connect(host=self.hive_host, port=self.hive_port,
                                username=self.hive_user, password=self.hive_pwd, 
                                auth='LDAP')

            if query_type == 'select':
                results = pd.read_sql(sql, conn)
                return results
            elif query_type in ['create', 'drop']:
                cursor = conn.cursor()
                cursor.execute(sql)
                results = cursor.fetch_logs()
                
                return results
            else:
                raise Exception("Error  : Connot regonize this query type, try again.")
        except Exception as e:
            msg = repr(e)
            print(msg)

        finally:
            conn.close()


    def _execute_by_presto(self, sql, query_type='select', columns=None):
        """ 通过presto执行sql
        :param sql         : 需要执行的sql
        :param query_type  : query的类型
        :param columns     : 是否使用手动给定的columns，因为presto返回的是个默认的二维数组
        :return:
        """
        try:
            conn = prestodb.dbapi.connect(
                        host=self.presto_host,
                        port=self.presto_port,
                        user=self.presto_user,
                        catalog=self.presto_schema,
                        schema=self.presto_schema)
            cursor = conn.cursor()
            cursor.execute(sql)
            if query_type == 'select':
                results = cursor.fetchall()
                if columns is None:
                    return pd.DataFrame(results)
                elif results and len(columns) == len(results[0]):
                    return pd.DataFrame(results, columns=columns)
                else:
                    return pd.DataFrame(results) 

            elif query_type in ['create', 'drop', 'desc']:
                results = cursor.fetchall()
                return results

            else:
                raise Exception("Error  : Connot regonize this query type, try again.")

        except Exception as e:
            msg = repr(e)
            print(msg)

        finally:
            conn.commit()
            conn.close()


    def execute(self, sql, query_type='select', columns=None, engine='hive'):
        """ 执行给定的sql
        :param sql         : 需要执行的sql
        :param query_type  : query的类型
        :param columns     : 是否使用手动给定的columns，因为presto返回的是个默认的二维数组
        :param engine      : 执行sql的引擎，目前可选为hive和presto
        :return:
        """
        if engine == 'hive':
            return self._execute_by_pyhive(sql, query_type)
        elif engine == 'presto':
            return self._execute_by_presto(sql, query_type, columns)
        else:
            raise Exception("Error  : Connot support this type engine, please check that and try again.")


    def download_table(self, library, table, overwrite_file=False, out_path='Downloads',  
        out_type='csv', delimiter='\x01', encoding='utf-8', **kwargs):
        """ 下载表，仅止支持下载Hive的表
        :param library         : 库名
        :param table           : 表名
        :param overwrite_file  : 如果本地文件已经存在，是否覆盖重写
        :param out_path        : 输出路径
        :param out_type        : 输出文件类型
        :param delimiter       : 输出文件如果是csv或xlsx的分隔符
        :param encoding        : 输出文件如果是csv或xlsx的编码模式
        :return:
        """
        if not os.path.exists(out_path) and out_path:
            os.mkdir(out_path)

        if not out_path:
            out_path = '.'

        table = table.lower()        
        out_path = f'{out_path}/{table}.{out_type}'


        if os.path.exists(out_path):
            print('INFO  : File Exists!')
            if not overwrite_file:
                try:
                    if out_type == 'csv':
                        return pd.read_csv(out_path, encoding=encoding)
                    elif out_type == 'xlsx':
                        return pd.read_excel(out_path, encoding=encoding)
                    elif out_type == 'pkl':
                        return pd.read_pickle(out_path)
                    else:
                        raise Exception("Error  : Connot regonize this type of file, please check that and try again.")
                except:
                    print('ERROR  : Something errors when readding the existed file.')
            else:
                print(f'INFO  : Remove existed .{out_type} file')
                os.remove(out_path)

        select_sql = f'SELECT * FROM {library}.{table}'
        results = self._execute_by_pyhive(select_sql)
        pos = len(table) + 1
        results.columns = [col[pos:] for col in results.columns]
        if type(results) == pd.DataFrame and results.shape[0] > 0:
            results = results.apply(pd.to_numeric, errors='ignore')
        else:
            print("ERROR  : The table may be embty or doesn't exists, please check that.")
            return 

        if out_type == 'csv':
            results.to_csv(out_path, index=False, encoding=encoding)
        elif out_type == 'xlsx':
            results.to_excel(out_path, index=False, encoding=encoding)
        elif out_type == 'pkl':
            results.to_pickle(out_path)
        else:
            print('ERROR  : Output file type must be in (csv, xlsx, pkl).')
        return results
    

    def create_table(self, library, table, file_path, overwrite_table=False, 
        delimiter='\x01', sep=',', encoding=None):
        """ 根据本地文件结构，在Hive上创建一张表
        :param library         : 库名
        :param table           : 表名
        :param file_path       : 文件路径，可以为df
        :param overwrite_table : 如果表已经存在，是否覆盖重写
        :param delimiter       : 表的分隔符
        :param sep             : csv / xlsx 文件的分隔符
        :param encoding        : csv / xlsx 文件的编码模式
        :return:
        """
        if type(file_path) == pd.DataFrame:
            df = file_path.copy()
        elif file_path.strip().endswith('.csv'):
            df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        elif file_path.strip().endswith('.xlsx'):
            df = pd.read_excel(file_path, sep=sep, encoding=encoding)
        elif file_path.strip().endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            raise Exception('INFO  : Input file format must be in (csv, xlsx, pkl).\nCannot read this file!')

        table_name = f'{library}.{table}'

        if overwrite_table:
            print('############## START OF DROP TABLE ##############')
            drop_sql = f'DROP TABLE IF EXISTS {table_name}'
            results = self._execute_by_pyhive(drop_sql, query_type='drop')
            # TODO : 需要检查是否成功
            for line in results:
                print(line)
            print('############## END OF DROP TABLE ##############\n\n')

        create_sql = self._gen_create_sql(df, table_name, delimiter)
        
        print('############## START OF CREATE TABLE ##############')
        print(create_sql)
        print('############## END OF CREATE TABLE ##############\n\n')
        results = self._execute_by_pyhive(create_sql, query_type='create')

        # TODO : 需要检查是否成功
        for line in results:
            print(line)

        for i in range(20000):
            tmp_file = f'00000_{i}'
            if not os.path.exists(tmp_file):
                break
        df.to_csv(tmp_file, index=False, header=False, sep=delimiter)

        try:
            table_path = f'{self.hdfs_root}/{library}.db/{table}'
            cmd = f'hdfs dfs -put {tmp_file} {table_path}'
            print('INFO  :', cmd)
            results = os.popen(cmd).readlines()
            for line in results:
                print('INFO  :', line)    
        except Exception as e:
            msg = repr(e)
            print(msg)
        finally:    
            os.remove(tmp_file)


    def _gen_create_sql(self, df, table_name, delimiter):
        """ 根据df数据，生成对应的CREATE TABLE语句
        :param df :
        :param table_name :
        :param delimiter :
        :return:
        """
        dtypes = self._get_dtypes(df)
        columns = ', '.join(dtypes['describe'])
        return f"CREATE TABLE IF NOT EXISTS {table_name} ( {columns} ) ROW FORMAT DELIMITED FIELDS TERMINATED BY '{delimiter}' STORED AS TEXTFILE"


    def _get_dtypes(self, df):
        """ 根据df数据，解析出对应的Hive格式
        :param df :
        :return:
        """
        dtypes_map = {'float64' : 'DECIMAL(20, 12)',
                      'int64'   : 'BIGINT',
                      'object'  : 'STRING'}
        dtypes = df.apply(pd.to_numeric, errors='ignore').dtypes.reset_index()
        dtypes.columns = ['name', 'dtype']
        dtypes['hive_type'] = dtypes['dtype'].map(lambda x : dtypes_map.get(str(x), 'STRING'))
        dtypes['describe'] = '`' + dtypes['name'] + '` ' + dtypes['hive_type']
        return dtypes

