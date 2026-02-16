import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置路径
WORKING_DIR = "./custom_kg"
TXT_PATH = "/Users/dongqing/Documents/phd/LightRAG/examples/edges_with_keywords_replaced2ndRoundKG_UPDATED.txt"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

def parse_kg_from_txt(txt_path):
    """
    从 TXT 文件中解析知识图谱数据（包含关键词的关系）
    
    返回格式：
    {
        "entities": [{"entity_name": ..., "entity_type": ..., "description": ..., "source_id": ...}],
        "relationships": [{"src_id": ..., "tgt_id": ..., "description": ..., "keywords": ..., "weight": ..., "source_id": ...}],
        "chunks": [{"content": ..., "source_id": ..., "source_chunk_index": ...}]
    }
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    entities = []
    relationships = []
    chunks = []
    
    # 找到所有 chunk 分隔符
    chunk_pattern = r'No\.: (\d+) of all the chunks'
    matches = list(re.finditer(chunk_pattern, full_text))
    
    logger.info(f"找到 {len(matches)} 个 chunks")
    
    # 处理每个 chunk
    for idx, match in enumerate(matches):
        chunk_id = match.group(1)
        source_id = f"chunk_{chunk_id}"
        
        # 确定 chunk 内容范围
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        
        # 提取 chunk 文本
        chunk_text = full_text[start_pos:end_pos].strip()
        
        if not chunk_text:
            logger.warning(f"Chunk {chunk_id} 内容为空，跳过")
            continue
        
        # 提取原始文本
        original_text_match = re.search(
            r'original text:\s*(.*?)(?=total tokens:|deepseek-v3 output:|$)', 
            chunk_text, 
            re.DOTALL
        )
        
        if original_text_match:
            original_text = original_text_match.group(1).strip()
            chunks.append({
                "content": original_text,
                "source_id": source_id,
                "source_chunk_index": int(chunk_id),
            })
            logger.info(f"  ✓ Chunk {chunk_id}: {len(original_text)} 字符")
        else:
            logger.warning(f"  ✗ Chunk {chunk_id}: 未找到原始文本")
        
        # 提取实体
        entity_pattern = r'\("entity"\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(.*?)\s*\)(?=\s*$|\s*\()'
        entity_matches = re.findall(entity_pattern, chunk_text, re.MULTILINE)
        
        for entity_name, entity_type, description in entity_matches:
            entities.append({
                "entity_name": entity_name.strip(),
                "entity_type": entity_type.strip(),
                "description": description.strip(),
                "source_id": source_id,
            })
        
        # 提取关系（带关键词）
        # 格式: ("relationship"|SRC|TGT|DESCRIPTION|WEIGHT|KEYWORDS)
        relation_pattern = r'\("relationship"\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(\d+)\s*\|\s*([^)]*?)\s*\)'
        relation_matches = re.findall(relation_pattern, chunk_text, re.MULTILINE)
        
        for src_id, tgt_id, description, weight, keywords in relation_matches:
            relationships.append({
                "src_id": src_id.strip(),
                "tgt_id": tgt_id.strip(),
                "description": description.strip(),
                "keywords": keywords.strip(),
                "weight": float(weight),
                "source_id": source_id,
            })
    
    logger.info(f"解析完成:")
    logger.info(f"  - 实体数量: {len(entities)}")
    logger.info(f"  - 关系数量: {len(relationships)}")
    logger.info(f"  - 文本块数量: {len(chunks)}")
    
    # 验证数据完整性
    validate_kg_data(entities, relationships, chunks)
    
    return {
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
    }

def validate_kg_data(entities, relationships, chunks):
    """
    验证 KG 数据的完整性
    """
    # 检查实体
    entity_names = set()
    for entity in entities:
        if not entity.get("entity_name"):
            logger.error(f"发现空实体名称: {entity}")
            raise ValueError("Entity name cannot be empty")
        if not entity.get("entity_type"):
            logger.error(f"发现空实体类型: {entity}")
            raise ValueError("Entity type cannot be empty")
        entity_names.add(entity["entity_name"])
    
    logger.info(f"  ✓ 实体验证通过: {len(entity_names)} 个唯一实体")
    
    # 检查关系
    missing_entities = set()
    for rel in relationships:
        if not rel.get("src_id"):
            logger.error(f"发现空源节点: {rel}")
            raise ValueError("Source ID cannot be empty")
        if not rel.get("tgt_id"):
            logger.error(f"发现空目标节点: {rel}")
            raise ValueError("Target ID cannot be empty")
        if not rel.get("keywords"):
            logger.warning(f"关系缺少关键词: {rel['src_id']} -> {rel['tgt_id']}")
        
        # 检查关系引用的实体是否存在
        if rel["src_id"] not in entity_names:
            missing_entities.add(rel["src_id"])
        if rel["tgt_id"] not in entity_names:
            missing_entities.add(rel["tgt_id"])
    
    if missing_entities:
        logger.warning(f"  ⚠ 关系引用了 {len(missing_entities)} 个未定义的实体:")
        for entity in list(missing_entities)[:10]:  # 只显示前10个
            logger.warning(f"    - {entity}")
        if len(missing_entities) > 10:
            logger.warning(f"    ... 还有 {len(missing_entities) - 10} 个")
    else:
        logger.info(f"  ✓ 关系验证通过: 所有实体都已定义")
    
    # 检查文本块
    if not chunks:
        logger.error("没有找到任何文本块")
        raise ValueError("No chunks found")
    
    logger.info(f"  ✓ 文本块验证通过: {len(chunks)} 个文本块")

def insert_kg_to_lightrag(kg_data, working_dir):
    """
    将解析的 KG 数据插入到 LightRAG
    """
    logger.info("\n初始化 LightRAG...")
    
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    
    logger.info("✓ LightRAG 初始化完成")
    
    logger.info("\n插入知识图谱...")
    rag.insert_custom_kg(kg_data)
    logger.info("✓ 知识图谱插入完成")
    
    # 输出存储信息
    logger.info("\n存储信息:")
    logger.info(f"  - 图数据库: {type(rag.chunk_entity_relation_graph).__name__}")
    logger.info(f"  - 实体向量DB: {type(rag.entities_vdb).__name__}")
    logger.info(f"  - 关系向量DB: {type(rag.relationships_vdb).__name__}")
    logger.info(f"  - 文本块向量DB: {type(rag.chunks_vdb).__name__}")
    logger.info(f"  - 文本块KV存储: {type(rag.text_chunks).__name__}")
    
    return rag

if __name__ == "__main__":
    try:
        # 步骤 1: 解析 TXT 文件
        print("\n[步骤 1/2] 解析 TXT 文件...")
        kg_data = parse_kg_from_txt(TXT_PATH)
        
        # 显示统计信息
        print(f"\n解析结果:")
        print(f"  ├─ 实体: {len(kg_data['entities'])} 个")
        print(f"  ├─ 关系: {len(kg_data['relationships'])} 个")
        print(f"  └─ 文本块: {len(kg_data['chunks'])} 个")
        
        
        # 步骤 2: 插入到 LightRAG
        print("\n[步骤 2/2] 插入到 LightRAG...")
        rag = insert_kg_to_lightrag(kg_data, WORKING_DIR)
        
    except Exception as e:
        logger.error(f"\n❌ 导入失败: {e}", exc_info=True)
        raise