import pytest

from sql_safety import validate_read_only_sql

ALLOWED = [
    "SELECT COUNT(*) FROM [Structure_2025].[dbo].[Payment]",
    "SELECT COUNT(*) FROM [Structure_2025].[dbo].[Payment];",
    "select top 10 * from dbo.WaterUser",
    "WITH paid AS (SELECT WuaID, SUM(PaymentMoney) AS total FROM dbo.Payment GROUP BY WuaID) "
    "SELECT * FROM paid ORDER BY total DESC",
    # keywords inside strings, quoted identifiers and comments are not code
    "SELECT * FROM dbo.WaterUser WHERE Status = 'in use'",
    "SELECT * FROM dbo.WaterUser WHERE Note = 'paid; see update from May'",
    "SELECT * FROM dbo.WaterUser WHERE Name = N'O''Brien -- not a comment'",
    "SELECT [Update], [Set] FROM dbo.AuditLog",
    'SELECT "Delete" FROM dbo.AuditLog',
    "SELECT PaymentMoney -- drop the nulls later\nFROM dbo.Payment",
    "SELECT PaymentMoney /* exec later */ FROM dbo.Payment",
    # column names that contain a blocked word are fine
    "SELECT LastUpdate, CreatedAt, Settings, IsDeleted FROM dbo.Payment",
    "SELECT CASE WHEN PaymentMoney > 0 THEN 1 ELSE 0 END AS Paid FROM dbo.Payment",
    "SELECT * FROM dbo.Payment ORDER BY PaymentDate OFFSET 0 ROWS FETCH NEXT 50 ROWS ONLY",
    "SELECT @@VERSION",
]

BLOCKED = [
    "",
    "   ",
    "-- only a comment",
    "UPDATE dbo.Payment SET PaymentMoney = 0",
    "DELETE FROM dbo.Payment",
    "EXEC xp_cmdshell 'dir'",
    "SELECT 1; DROP TABLE dbo.Payment",
    # T-SQL runs a second statement without a semicolon
    "SELECT 1 SHUTDOWN WITH NOWAIT",
    "SELECT 1 KILL 55",
    "SELECT 1 WAITFOR DELAY '02:00:00'",
    "SELECT 1 DISABLE TRIGGER ALL ON DATABASE",
    "SELECT 1 DECLARE @s nvarchar(99) = 'x' EXEC(@s)",
    "SELECT 1 SET ROWCOUNT 0",
    "SELECT 1 WHILE 1 = 1 SELECT 2",
    "SELECT 1 IF 1 = 1 SELECT 2",
    "SELECT 1 RECONFIGURE",
    "SELECT 1 BEGIN TRAN",
    # reaching outside the database
    "SELECT * FROM OPENROWSET(BULK 'C:\\Windows\\win.ini', SINGLE_CLOB) AS x",
    "SELECT * FROM OPENQUERY(LinkedServer, 'SELECT 1')",
    "SELECT * FROM OPENDATASOURCE('SQLNCLI', 'Data Source=x').db.dbo.t",
    "SELECT * INTO dbo.PaymentCopy FROM dbo.Payment",
    "SELECT * FROM sys.sp_helpdb",
    "WITH x AS (SELECT 1 AS a) DELETE FROM dbo.Payment",
    # a statement must not hide behind string or comment tricks
    "SELECT 'it''s' SHUTDOWN",
    "SELECT 1 /* /* nested */ still a comment */ SHUTDOWN",
    "SELECT 1 /* /* */ ' */ SHUTDOWN --'",
    "SELECT [a]]b] SHUTDOWN",
    # unterminated input is rejected, not guessed at
    "SELECT 'abc",
    "SELECT 1 /* open",
    "SELECT [abc",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only_queries(sql):
    is_safe, message = validate_read_only_sql(sql)
    assert is_safe, message


@pytest.mark.parametrize("sql", BLOCKED)
def test_blocks_anything_else(sql):
    is_safe, _ = validate_read_only_sql(sql)
    assert not is_safe


def test_none_is_rejected():
    assert validate_read_only_sql(None)[0] is False
