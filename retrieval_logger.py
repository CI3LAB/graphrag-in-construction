import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """存储单次检索的完整结果"""
    
    # 基本信息
    query: str
    query_mode: str  # local, global, hybrid, mix, naive
    timestamp: float
    
    # 检索到的内容
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    text_chunks: List[Dict[str, Any]]
    
    # 关键词信息
    high_level_keywords: List[str]
    low_level_keywords: List[str]
    
    # 检索元数据
    metadata: Dict[str, Any]
    
    # 最终回答（可选）
    final_response: Optional[str] = None


class RetrievalLogger:
    """检索结果日志记录器"""
    
    def __init__(self, log_dir: str = "./retrieval_logs"):
        """
        初始化检索日志记录器
        
        Args:
            log_dir: 日志存储目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建当前会话的日志文件
        self.session_id = int(time.time())
        self.log_file = self.log_dir / f"retrieval_log_{self.session_id}.jsonl"
        
        logger.info(f"RetrievalLogger initialized. Log file: {self.log_file}")
    
    def log_retrieval(self, retrieval_result: RetrievalResult) -> None:
        """
        记录单次检索结果到JSONL文件
        
        Args:
            retrieval_result: 检索结果对象
        """
        try:
            # 转换为字典
            result_dict = asdict(retrieval_result)
            
            # 写入JSONL文件（每行一个JSON对象）
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False)
                f.write('\n')
            
            logger.debug(f"Logged retrieval result for query: {retrieval_result.query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to log retrieval result: {e}")
    
    def load_logs(self, log_file: Optional[str] = None) -> List[RetrievalResult]:
        """
        加载日志文件中的检索结果
        
        Args:
            log_file: 指定日志文件路径，None则使用当前会话的日志
            
        Returns:
            检索结果列表
        """
        file_path = Path(log_file) if log_file else self.log_file
        
        if not file_path.exists():
            logger.warning(f"Log file not found: {file_path}")
            return []
        
        results = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result_dict = json.loads(line)
                        results.append(RetrievalResult(**result_dict))
            
            logger.info(f"Loaded {len(results)} retrieval results from {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load logs from {file_path}: {e}")
            return []
    def export_to_json(self, output_file: str, log_file: Optional[str] = None) -> None:
        """
        导出日志为单个JSON文件（便于分析）
        
        Args:
            output_file: 输出文件路径
            log_file: 源日志文件，None则导出整个日志目录的所有文件
        """
        # 🔥 新增：如果没有指定 log_file，则读取整个目录的所有 JSONL 文件
        if log_file is None:
            all_results = []
            
            # 获取目录中所有 .jsonl 文件
            jsonl_files = sorted(self.log_dir.glob("retrieval_log_*.jsonl"))
            
            if not jsonl_files:
                logger.warning(f"No JSONL files found in {self.log_dir}")
                return
            
            logger.info(f"Found {len(jsonl_files)} JSONL files to export")
            
            # 依次读取每个 JSONL 文件
            for jsonl_file in jsonl_files:
                logger.info(f"Reading {jsonl_file.name}...")
                results = self.load_logs(str(jsonl_file))
                all_results.extend(results)
            
            logger.info(f"Total loaded: {len(all_results)} retrieval results")
            
        else:
            # 如果指定了 log_file，只读取该文件（保持原有功能）
            all_results = self.load_logs(log_file)
        
        try:
            # 🔥 确保输出文件的父目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Exported {len(all_results)} results to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export to {output_file}: {e}")    
    
    def get_statistics(self, log_file: Optional[str] = None) -> Dict[str, Any]:
        """
        获取检索日志的统计信息
        
        Args:
            log_file: 指定日志文件，None则使用当前会话的日志
            
        Returns:
            统计信息字典
        """
        results = self.load_logs(log_file)
        
        if not results:
            return {}
        
        stats = {
            "total_queries": len(results),
            "mode_distribution": {},
            "avg_entities_per_query": 0,
            "avg_relationships_per_query": 0,
            "avg_chunks_per_query": 0,
            "queries_with_no_results": 0,
        }
        
        total_entities = 0
        total_relationships = 0
        total_chunks = 0
        
        for result in results:
            # 模式分布
            mode = result.query_mode
            stats["mode_distribution"][mode] = stats["mode_distribution"].get(mode, 0) + 1
            
            # 累计数量
            total_entities += len(result.entities)
            total_relationships += len(result.relationships)
            total_chunks += len(result.text_chunks)
            
            # 空结果查询
            if (not result.entities and not result.relationships and not result.text_chunks):
                stats["queries_with_no_results"] += 1
        
        # 计算平均值
        stats["avg_entities_per_query"] = total_entities / len(results)
        stats["avg_relationships_per_query"] = total_relationships / len(results)
        stats["avg_chunks_per_query"] = total_chunks / len(results)
        
        return stats