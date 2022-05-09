def find_tables(searchTable, dbaseList=None):
  #function to find name of databases and tables with column name specified
  #returns all accessible dbase objects in list supplied in call to function.tables.columns with column matching provided column name
  
  #create empty list to dump objects since dataframes are immutible (arrgh)
  dumpList = []
  
  #create list of column names
  outColList = ['dbase','table']
  
  #check if dbase list supplied
  if dbaseList == None:
    
    #indicator for later
    dbaseNone = 1
    
    #get dbase names
    dbases = spark.sql('show databases')
    countDbases = dbases.count()
    dbaseList = dbases.take(countDbases)
    
  else:
    #indicator for later
    dbaseNone = 0    

  #lop thru dbases
  for dbase in dbaseList:
    dbase = str(dbase)
    if dbaseNone == 1:
      dbase = dbase[18:-2]
    
    #get table names
    tables = spark.sql('show tables in ' + dbase + '')
    countTables = tables.count()
    tableList = tables.drop("database", "isTemporary")
    tableList = tableList.take(countTables)

    #set counter to pull in elements from dataTyepList
    x=0
    
    #loop thru tables
    for table in tableList:
      table = str(table)
      table = table[15:-2]
      if searchTable.upper() in table.upper():
        dumpList.append([dbase,table])
        
      x =+ 1
   
  #create final dataframe object to return
  returnDataFrame = spark.createDataFrame(dumpList,outColList)   
  return returnDataFrame
