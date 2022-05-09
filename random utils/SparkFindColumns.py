def find_columns(searchColumn, dbaseList):
  #function to find name of databases and tables with column name specified
  #returns all accessible dbase objects in list supplied in call to function.tables.columns with column matching provided column name
  
  #create empty list to dump objects since dataframes are immutible (arrgh)
  dumpList = []
  
  #create list of column names
  outColList = ['dbase','table','column_name','data_type']
  
  #get dbase names  - this is if you wanted to look in all dbases, problem is not all tables are readable and it causes issues down stream...
  #dbases = spark.sql('show databases')
  #countDbases = dbases.count()
  #dbaseList = dbases.take(countDbases)

  #lop thru dbases
  for dbase in dbaseList:
    dbase = str(dbase)
    #dbase = dbase[15:-2]
    
    #get table names
    tables = spark.sql('show tables in ' + dbase + '')
    countTables = tables.count()
    tableList = tables.drop("database", "isTemporary")
    tableList = tableList.take(countTables)

    #loop thru tables
    for table in tableList:
      table = str(table)
      table = table[15:-2]
      
      #get column names
      columns = spark.sql('describe ' + dbase + '.' + table + '')
      countColumns = columns.count()
      columnList = columns.select('col_name')
      datatypeList = columns.select('data_type')
      columnList = columnList.take(countColumns)
      datatypeList = datatypeList.take(countColumns)
      
      #set counter to pull in elements from dataTyepList
      x=0
      
      #loop thru columns
      for colu in columnList:
        colu = str(colu)
        columnName = colu[14:-2]
        if searchColumn.upper() in columnName.upper():
          dtype = datatypeList[x]     
          dumpList.append([dbase,table,columnName,dtype])
        
        x =+ 1
        
    
  #create final dataframe object to return
  returnDataFrame = spark.createDataFrame(dumpList,outColList)   
  return returnDataFrame
